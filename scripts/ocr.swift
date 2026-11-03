// Reads text in images with Apple Vision. Usage: ocr IMAGE...
// Languages come from OCR_LANGS (comma-separated, e.g. "en-US,th-TH").
// Prints one JSON object per image on its own line:
//   {"path": "...", "items": [{"text": "...", "conf": 0.98, "box": [x, y, w, h]}]}
// Boxes are fractions of the image with the origin at the TOP-left.
import Foundation
import ImageIO
import Vision

let languages = (ProcessInfo.processInfo.environment["OCR_LANGS"] ?? "en-US")
    .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }

func round3(_ v: CGFloat) -> Double { (Double(v) * 1000).rounded() / 1000 }

func readText(_ path: String) -> [[String: Any]] {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return [] }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = languages
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.012
    do {
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    } catch {
        return []
    }
    var found: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let best = observation.topCandidates(1).first, best.confidence >= 0.3 else { continue }
        let r = observation.boundingBox  // Vision measures from the bottom-left
        found.append([
            "text": best.string,
            "conf": Double(best.confidence),
            "box": [round3(r.minX), round3(1 - r.maxY), round3(r.width), round3(r.height)],
        ])
    }
    return found
}

for path in CommandLine.arguments.dropFirst() {
    let row: [String: Any] = ["path": path, "items": readText(path)]
    if let data = try? JSONSerialization.data(withJSONObject: row),
       let line = String(data: data, encoding: .utf8) {
        print(line)
    }
}
