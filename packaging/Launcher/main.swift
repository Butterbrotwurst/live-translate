// LiveTranslate launcher: runs the first-run setup with a progress window,
// then starts the local server and opens the UI in the browser.
import AppKit
import AVFoundation

let kPortRange = 8765...8790
let appSupport = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/LiveTranslate")

// MARK: - line-buffered pipe reader

final class LineReader {
    private var buffer = Data()
    private let onLine: (String) -> Void
    init(_ pipe: Pipe, onLine: @escaping (String) -> Void) {
        self.onLine = onLine
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else { return }
            self?.feed(chunk)
        }
    }
    private func feed(_ data: Data) {
        buffer.append(data)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = buffer[buffer.startIndex..<nl]
            buffer.removeSubrange(buffer.startIndex...nl)
            if let s = String(data: line, encoding: .utf8) {
                DispatchQueue.main.async { self.onLine(s) }
            }
        }
    }
}

// MARK: - window

final class Controller: NSObject, NSApplicationDelegate {
    let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 560, height: 180),
        styleMask: [.titled, .closable, .miniaturizable],
        backing: .buffered, defer: false)

    let title = NSTextField(labelWithString: "Live Translate")
    let status = NSTextField(labelWithString: "Wird gestartet …")
    let bar = NSProgressIndicator()
    let detailsToggle = NSButton()
    let logView = NSTextView()
    let logScroll = NSScrollView()
    let actionButton = NSButton()

    var setupTask: Process?
    var serverTask: Process?
    var port = kPortRange.lowerBound
    var running = false
    var stepBase = 0.0      // progress already completed by finished steps
    var stepSpan = 1.0      // share of the bar owned by the current step
    private var readers: [LineReader] = []

    // MARK: layout

    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMenu()
        buildWindow()
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
        startSetup()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ note: Notification) { stopServer() }

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Über Live Translate", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Live Translate beenden", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)
        NSApp.mainMenu = main
    }

    private func buildWindow() {
        window.title = "Live Translate"
        window.center()
        window.isReleasedWhenClosed = false

        title.font = .systemFont(ofSize: 17, weight: .semibold)
        status.font = .systemFont(ofSize: 13)
        status.textColor = .secondaryLabelColor
        status.lineBreakMode = .byTruncatingTail

        bar.isIndeterminate = false
        bar.minValue = 0
        bar.maxValue = 1
        bar.doubleValue = 0
        bar.controlSize = .regular

        detailsToggle.title = "Details einblenden"
        detailsToggle.bezelStyle = .inline
        detailsToggle.isBordered = false
        detailsToggle.contentTintColor = .secondaryLabelColor
        detailsToggle.target = self
        detailsToggle.action = #selector(toggleDetails)

        logView.isEditable = false
        logView.font = .monospacedSystemFont(ofSize: 10, weight: .regular)
        logView.textColor = .secondaryLabelColor
        logView.drawsBackground = false
        logScroll.documentView = logView
        logScroll.hasVerticalScroller = true
        logScroll.drawsBackground = false
        logScroll.isHidden = true

        actionButton.title = "Im Browser öffnen"
        actionButton.bezelStyle = .rounded
        actionButton.keyEquivalent = "\r"
        actionButton.target = self
        actionButton.action = #selector(primaryAction)
        actionButton.isHidden = true

        let head = NSStackView(views: [title, status, bar])
        head.orientation = .vertical
        head.alignment = .leading
        head.spacing = 6
        for v in [title, status, bar] { v.translatesAutoresizingMaskIntoConstraints = false }

        let footer = NSStackView(views: [detailsToggle, NSView(), actionButton])
        footer.orientation = .horizontal

        let root = NSStackView(views: [head, logScroll, footer])
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 12
        root.edgeInsets = NSEdgeInsets(top: 20, left: 22, bottom: 18, right: 22)
        root.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.addSubview(root)
        NSLayoutConstraint.activate([
            root.topAnchor.constraint(equalTo: content.topAnchor),
            root.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            root.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            root.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            head.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -44),
            bar.widthAnchor.constraint(equalTo: head.widthAnchor),
            footer.widthAnchor.constraint(equalTo: head.widthAnchor),
            logScroll.widthAnchor.constraint(equalTo: head.widthAnchor),
            logScroll.heightAnchor.constraint(equalToConstant: 150),
        ])
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func toggleDetails() {
        logScroll.isHidden.toggle()
        detailsToggle.title = logScroll.isHidden ? "Details einblenden" : "Details ausblenden"
        window.setContentSize(NSSize(width: 560, height: logScroll.isHidden ? 180 : 340))
    }

    @objc private func primaryAction() {
        if running { openBrowser() } else { startSetup() }
    }

    private func log(_ line: String) {
        logView.textStorage?.append(NSAttributedString(
            string: line + "\n",
            attributes: [.font: NSFont.monospacedSystemFont(ofSize: 10, weight: .regular),
                         .foregroundColor: NSColor.secondaryLabelColor]))
        logView.scrollToEndOfDocument(nil)
    }

    // MARK: setup

    func startSetup() {
        actionButton.isHidden = true
        status.stringValue = "Einrichtung wird geprüft …"
        bar.doubleValue = 0
        let res = Bundle.main.resourcePath!
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = [res + "/bootstrap.sh", res]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        readers.append(LineReader(pipe) { [weak self] in self?.handle($0) })
        task.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self, !self.running else { return }
                if proc.terminationStatus != 0 { self.failed("Die Einrichtung wurde abgebrochen.") }
            }
        }
        setupTask = task
        do { try task.run() } catch { failed("Die Einrichtung konnte nicht gestartet werden: \(error.localizedDescription)") }
    }

    private func handle(_ line: String) {
        if line.hasPrefix("@@STEP ") {
            let rest = String(line.dropFirst(7))
            let parts = rest.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
            let frac = parts.first.map(String.init)?.split(separator: "/").compactMap { Double($0) } ?? []
            if frac.count == 2, frac[1] > 0 {
                stepSpan = 1.0 / frac[1]
                stepBase = (frac[0] - 1) * stepSpan
                bar.doubleValue = stepBase
            }
            if parts.count > 1 { status.stringValue = String(parts[1]) }
            log(line)
        } else if line.hasPrefix("@@PROG ") {
            if let v = Double(line.dropFirst(7)) {
                bar.doubleValue = min(stepBase + v * stepSpan, 1.0)
            }
        } else if line.hasPrefix("@@FAIL ") {
            failed(String(line.dropFirst(7)))
        } else if line.hasPrefix("@@READY") {
            bar.doubleValue = 1.0
            startServer()
        } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
            log(line)
        }
    }

    private func failed(_ message: String) {
        status.stringValue = message
        bar.doubleValue = 0
        actionButton.title = "Erneut versuchen"
        actionButton.isHidden = false
        if logScroll.isHidden { toggleDetails() }
    }

    // MARK: server

    private func freePort() -> Int {
        for p in kPortRange {
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { continue }
            var yes: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = UInt16(p).bigEndian
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")
            let ok = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
                }
            }
            close(fd)
            if ok { return p }
        }
        return kPortRange.lowerBound
    }

    func startServer() {
        status.stringValue = "Modelle werden geladen …"
        bar.isIndeterminate = true
        bar.startAnimation(nil)
        port = freePort()

        let res = Bundle.main.resourcePath!
        let src = appSupport.appendingPathComponent("src").path
        let task = Process()
        task.executableURL = URL(fileURLWithPath: res + "/bin/uv")
        task.arguments = ["run", "--directory", src, "--frozen",
                          "live-translate-ui", "--port", String(port), "--no-browser"]
        var env = ProcessInfo.processInfo.environment
        env["HF_HOME"] = appSupport.appendingPathComponent("hf").path
        env["UV_CACHE_DIR"] = appSupport.appendingPathComponent("uv-cache").path
        env["UV_PYTHON_INSTALL_DIR"] = appSupport.appendingPathComponent("python").path
        env["UV_NO_CONFIG"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        task.environment = env
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        readers.append(LineReader(pipe) { [weak self] in self?.log($0) })
        task.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.running else { return }
                self.running = false
                self.bar.stopAnimation(nil)
                self.bar.isIndeterminate = false
                self.failed("Der Server wurde beendet.")
            }
        }
        serverTask = task
        do { try task.run() } catch {
            failed("Der Server konnte nicht gestartet werden: \(error.localizedDescription)")
            return
        }
        waitForServer(attempt: 0)
    }

    private func waitForServer(attempt: Int) {
        guard attempt < 600 else { failed("Der Server antwortet nicht."); return }
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/")!)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { [weak self] _, response, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                if (response as? HTTPURLResponse)?.statusCode != nil {
                    self.serverIsUp()
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.waitForServer(attempt: attempt + 1)
                    }
                }
            }
        }.resume()
    }

    private func serverIsUp() {
        guard !running else { return }
        running = true
        bar.stopAnimation(nil)
        bar.isIndeterminate = false
        bar.doubleValue = 1
        status.stringValue = "Läuft auf 127.0.0.1:\(port) — dieses Fenster offen lassen."
        actionButton.title = "Im Browser öffnen"
        actionButton.isHidden = false
        openBrowser()
    }

    private func openBrowser() {
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(port)/")!)
    }

    func stopServer() {
        guard let task = serverTask, task.isRunning else { return }
        running = false
        task.terminate()
        let deadline = Date().addingTimeInterval(4)
        while task.isRunning && Date() < deadline { usleep(100_000) }
        if task.isRunning { kill(task.processIdentifier, SIGKILL) }
    }
}

let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.run()
