#!/usr/bin/env python3
import os
import sys
import json
import time
import uuid
import mimetypes
import argparse
import subprocess
import shutil
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse
import base64

PAPER_DIR = Path(__file__).resolve().parent
COMMENTS_FILE = PAPER_DIR / "comments.json"
INDEX_HTML_FILE = PAPER_DIR / "index.html"

def load_comments():
    if COMMENTS_FILE.exists():
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_comments(comments):
    try:
        with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving comments: {e}")
        return False


def is_safe_path(target_path: Path) -> bool:
    try:
        target_path.resolve().relative_to(PAPER_DIR.resolve())
        return True
    except ValueError:
        return False

def get_file_tree():
    ignore_exts = {".aux", ".fdb_latexmk", ".fls", ".out", ".synctex.gz", ".pyc"}
    ignore_dirs = {".git", "__pycache__", ".claude"}

    def scan_dir(dir_path: Path, rel_base=""):
        items = []
        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for entry in entries:
                if entry.name in ignore_dirs or entry.name.startswith("."):
                    continue
                
                rel_path = str(entry.relative_to(PAPER_DIR))
                
                if entry.is_dir():
                    children = scan_dir(entry, rel_path)
                    items.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "dir",
                        "children": children
                    })
                else:
                    ext = entry.suffix.lower()
                    if ext in ignore_exts and entry.name != "main.log":
                        continue
                    
                    is_text = ext in {".tex", ".bib", ".cls", ".bst", ".sty", ".txt", ".md", ".json", ".sh", ".py", ".log", ".bbl", ".blg"}
                    is_img = ext in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".eps"}
                    is_pdf = ext == ".pdf"
                    
                    try:
                        size = entry.stat().st_size
                        mtime = entry.stat().st_mtime
                    except Exception:
                        size = 0
                        mtime = 0

                    items.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "file",
                        "ext": ext,
                        "size": size,
                        "mtime": mtime,
                        "isText": is_text,
                        "isImage": is_img,
                        "isPdf": is_pdf
                    })
        except Exception as e:
            print(f"Error scanning dir {dir_path}: {e}")
        return items

    return scan_dir(PAPER_DIR)

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class LatexLiveHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
        elif path == "/pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_cors_headers()
            self.end_headers()
        else:
            super().do_HEAD()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_cors_headers()
            self.end_headers()
            if INDEX_HTML_FILE.exists():
                with open(INDEX_HTML_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"index.html not found.")
            return

        elif path == "/api/tree":
            tree = get_file_tree()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"tree": tree}).encode("utf-8"))
            return

        elif path == "/api/files":
            tex_files = sorted([str(f.relative_to(PAPER_DIR)) for f in PAPER_DIR.glob("**/*.tex") if not f.name.startswith(".")])
            bib_files = sorted([str(f.relative_to(PAPER_DIR)) for f in PAPER_DIR.glob("**/*.bib") if not f.name.startswith(".")])
            pdf_files = sorted([str(f.relative_to(PAPER_DIR)) for f in PAPER_DIR.glob("**/*.pdf") if not f.name.startswith(".")])
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"files": tex_files + bib_files, "pdfs": pdf_files}).encode("utf-8"))
            return

        elif path == "/api/file":
            rel_name = query.get("name", ["main.tex"])[0].lstrip("/")
            file_path = (PAPER_DIR / rel_name).resolve()
            if not is_safe_path(file_path) or not file_path.exists() or file_path.is_dir():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "File not found"}).encode("utf-8"))
                return

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"name": rel_name, "content": content}).encode("utf-8"))
            return

        elif path == "/api/raw":
            rel_name = query.get("path", [""])[0].lstrip("/")
            file_path = (PAPER_DIR / rel_name).resolve()
            if not is_safe_path(file_path) or not file_path.exists() or file_path.is_dir():
                self.send_error(404, "File not found")
                return

            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = "application/octet-stream"

            with open(file_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(data)
            return

        elif path == "/api/comments":
            comments = load_comments()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(comments).encode("utf-8"))
            return

        elif path == "/pdf":
            rel_name = query.get("file", ["main.pdf"])[0].lstrip("/")
            if not rel_name.endswith(".pdf"):
                rel_name += ".pdf"
            
            pdf_path = (PAPER_DIR / rel_name).resolve()
            if not is_safe_path(pdf_path):
                self.send_error(403, "Access denied")
                return

            if not pdf_path.exists():
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(b"PDF not found. Please click 'Recompile' first.")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_cors_headers()
            if "download" in query:
                self.send_header("Content-Disposition", f'attachment; filename="{pdf_path.name}"')
            else:
                self.send_header("Content-Disposition", f'inline; filename="{pdf_path.name}"')
            
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return

        elif path == "/log":
            log_path = PAPER_DIR / "main.log"
            if log_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                with open(log_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Log not found")
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"

        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if path == "/api/upload":
            folder = data.get("folder", "").strip().lstrip("/")
            files = data.get("files", [])

            target_dir = (PAPER_DIR / folder).resolve()
            if not is_safe_path(target_dir):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid target directory"}).encode("utf-8"))
                return

            target_dir.mkdir(parents=True, exist_ok=True)
            uploaded_list = []

            for f_item in files:
                raw_name = Path(f_item.get("name", "")).name
                b64_data = f_item.get("data", "")
                if not raw_name or not b64_data:
                    continue

                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]

                try:
                    file_bytes = base64.b64decode(b64_data)
                    dest = target_dir / raw_name
                    with open(dest, "wb") as f_out:
                        f_out.write(file_bytes)
                    uploaded_list.append(str(dest.relative_to(PAPER_DIR)))
                except Exception as e:
                    print(f"Error saving upload {raw_name}: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "uploaded": uploaded_list}).encode("utf-8"))
            return

        elif path == "/api/file/create":
            rel_path = data.get("path", "").strip().lstrip("/")
            item_type = data.get("type", "file") # "file" or "dir"

            if not rel_path:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Path cannot be empty"}).encode("utf-8"))
                return

            target_path = (PAPER_DIR / rel_path).resolve()
            if not is_safe_path(target_path):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid path"}).encode("utf-8"))
                return

            try:
                if item_type == "dir":
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not target_path.exists():
                        target_path.touch()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "path": rel_path}).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                return

        elif path == "/api/file/rename":
            old_rel = data.get("oldPath", "").strip().lstrip("/")
            new_rel = data.get("newPath", "").strip().lstrip("/")

            old_path = (PAPER_DIR / old_rel).resolve()
            new_path = (PAPER_DIR / new_rel).resolve()

            if not is_safe_path(old_path) or not is_safe_path(new_path) or not old_path.exists():
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid path or file does not exist"}).encode("utf-8"))
                return

            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "path": new_rel}).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                return

        elif path == "/api/file":
            file_name = data.get("name", "main.tex").strip().lstrip("/")
            content = data.get("content", "")

            file_path = (PAPER_DIR / file_name).resolve()
            if not is_safe_path(file_path):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid file path"}).encode("utf-8"))
                return

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        elif path == "/api/comments":
            comments = load_comments()
            new_comment = {
                "id": "c-" + str(uuid.uuid4())[:8],
                "author": data.get("author", "Advisor"),
                "section": data.get("section", ""),
                "page": data.get("page", ""),
                "quote": data.get("quote", ""),
                "type": data.get("type", "suggestion"),
                "text": data.get("text", ""),
                "resolved": False,
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            comments.insert(0, new_comment)
            save_comments(comments)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "comment": new_comment}).encode("utf-8"))
            return

        elif path == "/api/compile":
            target_file = data.get("file", "main.tex").strip().lstrip("/")
            engine = data.get("engine", "latexmk")

            target_path = (PAPER_DIR / target_file).resolve()
            if not is_safe_path(target_path) or not target_path.exists():
                target_file = "main.tex"
                target_path = PAPER_DIR / "main.tex"

            if engine == "xelatex":
                cmd = ["xelatex", "-interaction=nonstopmode", target_file]
            elif engine == "pdflatex":
                cmd = ["pdflatex", "-interaction=nonstopmode", target_file]
            else: # latexmk
                cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", target_file]

            start_t = time.time()
            res = subprocess.run(
                cmd,
                cwd=str(PAPER_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            duration = time.time() - start_t

            pdf_file = target_file.replace(".tex", ".pdf")
            has_pdf = (PAPER_DIR / pdf_file).exists()

            response_data = {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "duration": round(duration, 3),
                "stdout": res.stdout,
                "stderr": res.stderr,
                "pdf_file": pdf_file,
                "has_pdf": has_pdf
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def do_PATCH(self):
        self.handle_comment_update()

    def do_PUT(self):
        self.handle_comment_update()

    def handle_comment_update(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/comments":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            cid = data.get("id")
            resolved = data.get("resolved")

            comments = load_comments()
            for c in comments:
                if c.get("id") == cid:
                    if resolved is not None:
                        c["resolved"] = bool(resolved)
                    break
            save_comments(comments)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")
    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/api/comments":
            cid = query.get("id", [""])[0]
            comments = load_comments()
            comments = [c for c in comments if c.get("id") != cid]
            save_comments(comments)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        elif parsed.path == "/api/file":
            rel_path = query.get("path", [""])[0].strip().lstrip("/")
            if not rel_path or rel_path == "main.tex" or rel_path == "llncs.cls":
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Cannot delete root critical file"}).encode("utf-8"))
                return

            target_path = (PAPER_DIR / rel_path).resolve()
            if not is_safe_path(target_path) or not target_path.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "File not found"}).encode("utf-8"))
                return

            try:
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                return

        self.send_error(404, "Endpoint not found")

def main():
    parser = argparse.ArgumentParser(description="LaTeX Live Studio Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on (default 8088)")
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = ThreadingSimpleServer(server_address, LatexLiveHandler)
    print(f"=====================================================")
    print(f" 🚀 LaTeX Live Studio, Explorer & AI Copilot: http://localhost:{args.port}")
    print(f" 📁 Workspace: {PAPER_DIR}")
    print(f" 🤖 Claude CLI: {CLAUDE_BIN}")
    print(f" 💬 Comments file: {COMMENTS_FILE}")
    print(f" ⚡ Ready for Cloudflare Tunnel on port {args.port}")
    print(f"=====================================================")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
