"""
repo_analysis_tools.py — GitHub repository analysis tools.

Capabilities:
  - Clone repos via bridge (git clone)
  - Analyze structure, tech stack, LOC, dependencies
  - GitHub CLI operations (gh) via bridge
  - AST parsing via tree-sitter (optional dependency)

Usage:
    from app.tools.repo_analysis_tools import create_repo_analysis_tools
    tools = create_repo_analysis_tools("repo_analyst")
"""

import json
import logging
import re
import shlex
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

# Tech stack detection: filename → technology
_TECH_INDICATORS = {
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "requirements.txt": "Python",
    "pyproject.toml": "Python",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java (Gradle)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "CMakeLists.txt": "C/C++",
    "Makefile": "Make",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    ".github/workflows": "GitHub Actions",
    "terraform": "Terraform",
    ".env.example": "Dotenv",
}

_EXT_TO_LANG = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React", ".tsx": "React+TS", ".go": "Go", ".rs": "Rust",
    ".java": "Java", ".rb": "Ruby", ".php": "PHP", ".c": "C",
    ".cpp": "C++", ".h": "C/C++ Header", ".cs": "C#", ".swift": "Swift",
    ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".md": "Markdown",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON", ".toml": "TOML",
    ".xml": "XML", ".proto": "Protocol Buffers",
}


def create_repo_analysis_tools(agent_id: str) -> list:
    """Create repository analysis tools via bridge.

    Returns empty list if bridge is unavailable.
    """
    try:
        from app.bridge_client import get_bridge
        bridge = get_bridge(agent_id)
        if not bridge:
            return []
        if not bridge.is_available():
            logger.debug(f"repo_analysis_tools: bridge unavailable for {agent_id}")
            return []
    except Exception:
        return []

    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    # ── Tool definitions ──────────────────────────────────────────

    class _CloneRepoInput(BaseModel):
        repo_url: str = Field(
            description="Git repository URL (HTTPS or git@) or GitHub shorthand (owner/repo)"
        )
        shallow: bool = Field(
            default=True,
            description="Shallow clone (--depth 1) for faster analysis",
        )

    class CloneRepoTool(BaseTool):
        name: str = "clone_repo"
        description: str = (
            "Clone a Git repository to a temporary directory on the host. "
            "Returns the local path for further analysis."
        )
        args_schema: Type[BaseModel] = _CloneRepoInput

        def _run(self, repo_url: str, shallow: bool = True) -> str:
            # Expand GitHub shorthand
            if re.match(r"^[\w.-]+/[\w.-]+$", repo_url):
                repo_url = f"https://github.com/{repo_url}.git"

            # Determine target directory
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            target = f"/tmp/crewai-repos/{repo_name}"

            # Remove existing if present
            bridge.execute(["rm", "-rf", target])

            cmd = ["git", "clone"]
            if shallow:
                cmd.extend(["--depth", "1"])
            cmd.extend([repo_url, target])

            result = bridge.execute(cmd)
            if "error" in result:
                return f"Error cloning: {result.get('detail', result['error'])}"
            return f"Repository cloned to: {target}"

    class _AnalyzeStructureInput(BaseModel):
        repo_path: str = Field(description="Local path to the cloned repository")

    class AnalyzeRepoStructureTool(BaseTool):
        name: str = "analyze_repo_structure"
        description: str = (
            "Analyze repository structure: file tree, tech stack, LOC by language, "
            "dependencies, and project layout."
        )
        args_schema: Type[BaseModel] = _AnalyzeStructureInput

        def _run(self, repo_path: str) -> str:
            # Get file listing
            result = bridge.list_files(repo_path, "**/*", recursive=True)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"

            files = result.get("files", [])
            if not files:
                return "No files found in repository."

            # Analyze extensions
            ext_counts: dict[str, int] = {}
            tech_stack: set[str] = set()
            total_files = 0
            dirs: set[str] = set()

            for f in files:
                name = f.get("name", "") if isinstance(f, dict) else str(f)
                path = PurePosixPath(name)
                total_files += 1

                # Track directories
                if path.parent != PurePosixPath("."):
                    dirs.add(str(path.parent))

                # Extension counting
                ext = path.suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1

                # Tech stack detection
                basename = path.name
                for indicator, tech in _TECH_INDICATORS.items():
                    if basename == indicator or indicator in name:
                        tech_stack.add(tech)

            # Build language breakdown
            lang_counts: dict[str, int] = {}
            for ext, count in ext_counts.items():
                lang = _EXT_TO_LANG.get(ext, "")
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + count

            # Sort by count
            sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
            sorted_exts = sorted(ext_counts.items(), key=lambda x: -x[1])

            lines = [
                f"=== Repository Analysis: {repo_path.split('/')[-1]} ===",
                f"Total files: {total_files}",
                f"Directories: {len(dirs)}",
                f"\nTech Stack: {', '.join(sorted(tech_stack)) if tech_stack else 'Unknown'}",
                "\nLanguage Breakdown:",
            ]
            for lang, count in sorted_langs[:15]:
                pct = (count / total_files) * 100
                lines.append(f"  {lang}: {count} files ({pct:.1f}%)")

            # Top-level directories
            top_dirs = sorted({str(PurePosixPath(d).parts[0]) for d in dirs if d})[:20]
            lines.append(f"\nTop-level directories: {', '.join(top_dirs)}")

            return "\n".join(lines)

    class _RepoMetricsInput(BaseModel):
        repo_path: str = Field(description="Local path to the cloned repository")

    class RepoMetricsTool(BaseTool):
        name: str = "repo_metrics"
        description: str = (
            "Calculate quantitative repository metrics: lines of code, "
            "file sizes, dependency count, and project health indicators."
        )
        args_schema: Type[BaseModel] = _RepoMetricsInput

        def _run(self, repo_path: str) -> str:
            # Use cloc if available, otherwise wc -l.
            # shlex.quote neutralizes shell metacharacters in repo_path so an
            # agent-provided path cannot break out of the find argument.
            qpath = shlex.quote(repo_path)
            result = bridge.execute(["sh", "-c", f"find {qpath} -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.go' -o -name '*.rs' -o -name '*.java' -o -name '*.rb' | head -500 | xargs wc -l 2>/dev/null | tail -1"])
            if "error" in result:
                loc_total = "unknown"
            else:
                loc_total = result.get("stdout", "").strip()

            # Check for common config files
            checks = {
                "README": "README.md",
                "LICENSE": "LICENSE",
                "CI/CD": ".github/workflows",
                "Tests": "tests",
                "Docs": "docs",
                ".gitignore": ".gitignore",
            }
            health = {}
            for label, path in checks.items():
                check_result = bridge.execute(["test", "-e", f"{repo_path}/{path}"])
                health[label] = "error" not in check_result

            # Dependency count
            dep_count = self._count_deps(repo_path)

            lines = [
                f"=== Repository Metrics ===",
                f"Lines of code (source files): {loc_total}",
                f"Dependencies: {dep_count}",
                "\nProject Health:",
            ]
            for label, exists in health.items():
                icon = "+" if exists else "-"
                lines.append(f"  [{icon}] {label}")

            return "\n".join(lines)

        def _count_deps(self, repo_path: str) -> str:
            """Count dependencies from package files."""
            counts = []
            # shlex.quote neutralizes shell metacharacters so paths supplied
            # by agents cannot inject additional commands.
            qpath = shlex.quote(repo_path)
            # Python
            r = bridge.execute(["sh", "-c", f"wc -l < {qpath}/requirements.txt 2>/dev/null"])
            if "error" not in r and r.get("stdout", "").strip().isdigit():
                counts.append(f"Python: {r['stdout'].strip()}")
            # Node
            r = bridge.execute(["sh", "-c", f"cat {qpath}/package.json 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(len(d.get('dependencies',{{}})) + len(d.get('devDependencies',{{}})))\""])
            if "error" not in r and r.get("stdout", "").strip().isdigit():
                counts.append(f"Node: {r['stdout'].strip()}")
            return ", ".join(counts) if counts else "unknown"

    class _GitHubCLIInput(BaseModel):
        command: str = Field(
            description="GitHub CLI command (without 'gh' prefix). "
            "Examples: 'repo list', 'pr list', 'issue create --title ...'",
        )

    class GitHubCLITool(BaseTool):
        name: str = "github_cli"
        description: str = (
            "Execute GitHub CLI (gh) commands on the host. "
            "Use for creating repos, PRs, issues, releases, and more. "
            "Requires 'gh' to be installed and authenticated on the host."
        )
        args_schema: Type[BaseModel] = _GitHubCLIInput

        def _run(self, command: str) -> str:
            cmd_parts = ["gh"] + command.split()
            result = bridge.execute(cmd_parts)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            output = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if stderr and not output:
                return f"gh error: {stderr[:500]}"
            return output[:5000] if output else "Command executed (no output)."

    class _DiagramInput(BaseModel):
        repo_path: str = Field(description="Local path to the cloned repository")
        focus: str = Field(
            default="directories",
            description="Diagram focus: 'directories' (folder structure) or 'imports' (dependency graph)",
        )

    class GenerateArchitectureDiagramTool(BaseTool):
        name: str = "generate_architecture_diagram"
        description: str = (
            "Generate a text-based architecture diagram of the repository. "
            "Outputs DOT format (Graphviz) for directory structure or import graph."
        )
        args_schema: Type[BaseModel] = _DiagramInput

        def _run(self, repo_path: str, focus: str = "directories") -> str:
            result = bridge.list_files(repo_path, "**/*", recursive=True)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"

            files = result.get("files", [])
            repo_name = repo_path.split("/")[-1]

            if focus == "directories":
                # Build directory tree as DOT graph
                dirs: set[str] = set()
                for f in files:
                    name = f.get("name", "") if isinstance(f, dict) else str(f)
                    parts = PurePosixPath(name).parts
                    for i in range(1, min(len(parts), 4)):  # Max depth 3
                        dirs.add("/".join(parts[:i]))

                dot_lines = [f'digraph "{repo_name}" {{', '  rankdir=LR;', '  node [shape=folder];']
                edges: set[str] = set()
                for d in sorted(dirs):
                    parts = d.split("/")
                    label = parts[-1]
                    node_id = d.replace("/", "_").replace("-", "_").replace(".", "_")
                    dot_lines.append(f'  {node_id} [label="{label}"];')
                    if len(parts) > 1:
                        parent_id = "/".join(parts[:-1]).replace("/", "_").replace("-", "_").replace(".", "_")
                        edge = f"  {parent_id} -> {node_id};"
                        if edge not in edges:
                            edges.add(edge)
                            dot_lines.append(edge)

                dot_lines.append("}")
                return "\n".join(dot_lines)
            else:
                return "Import graph analysis requires tree-sitter (Phase 3 advanced feature)."

    return [
        CloneRepoTool(),
        AnalyzeRepoStructureTool(),
        RepoMetricsTool(),
        GitHubCLITool(),
        GenerateArchitectureDiagramTool(),
    ]
