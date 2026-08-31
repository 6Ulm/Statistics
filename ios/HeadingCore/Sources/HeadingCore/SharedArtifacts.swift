import Foundation

/// Locates the repository-root shared artifacts (`config/`, `schemas/`, `testdata/`) that
/// the Android runtime, the iOS runtime and `analysis/` all read.
///
/// SPEC.md §37.1 requires both platforms to consume the *same* files rather than copies,
/// because independently translated constants are a common source of false parity. In a
/// shipped app these files are bundled read-only; in the test runtime they are read from
/// the checkout, located by walking up from this source file.
public enum SharedArtifacts {

    public enum ArtifactError: Error, CustomStringConvertible {
        case repositoryRootNotFound(searchedFrom: String)
        case missingArtifact(String)

        public var description: String {
            switch self {
            case .repositoryRootNotFound(let from):
                return "could not locate the repository root walking up from \(from); a runtime "
                    + "that cannot find the shared artifacts must fail rather than fall back to a copy"
            case .missingArtifact(let path):
                return "required shared artifact is missing: \(path)"
            }
        }
    }

    /// Environment override, used by CI where the checkout is not the source location.
    public static let repoRootEnvironmentVariable = "FSC_REPO_ROOT"

    public static func repositoryRoot(from file: StaticString = #filePath) throws -> URL {
        if let override = ProcessInfo.processInfo.environment[repoRootEnvironmentVariable] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        var directory = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        let marker = "config/precision-profile-v1.json"
        for _ in 0..<12 {
            if FileManager.default.fileExists(atPath: directory.appendingPathComponent(marker).path) {
                return directory
            }
            directory = directory.deletingLastPathComponent()
        }
        throw ArtifactError.repositoryRootNotFound(searchedFrom: "\(file)")
    }

    public static func url(_ relativePath: String, from file: StaticString = #filePath) throws -> URL {
        let candidate = try repositoryRoot(from: file).appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            throw ArtifactError.missingArtifact(relativePath)
        }
        return candidate
    }

    public static func precisionProfileURL(from file: StaticString = #filePath) throws -> URL {
        try url("config/precision-profile-v1.json", from: file)
    }

    public static func fengShuiRuleSetURL(from file: StaticString = #filePath) throws -> URL {
        try url("config/feng-shui-rules-v1.json", from: file)
    }

    public static func gradeReachabilityClaimsURL(from file: StaticString = #filePath) throws -> URL {
        try url("testdata/grade-reachability-claims-v1.json", from: file)
    }

    public static func exampleEngineOutputEventURL(from file: StaticString = #filePath) throws -> URL {
        try url("testdata/telemetry-event-engine-output-v1.example.json", from: file)
    }
}
