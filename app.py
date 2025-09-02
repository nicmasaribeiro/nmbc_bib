from __future__ import annotations
import datetime
import mimetypes
from typing import Optional
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
import os, hashlib, datetime
from dataclasses import dataclass
from typing import List
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, abort, jsonify
)
from flask_login import (
    LoginManager, login_user, current_user, logout_user,
    login_required, UserMixin
)
from flask_socketio import SocketIO, join_room, leave_room, emit
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import (
    create_engine, func, select, ForeignKey, UniqueConstraint,
    case, Integer, String, DateTime, Text, Boolean
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, Session
)
from typing import List, Optional
from sqlalchemy.orm import sessionmaker
# add import
from sqlalchemy.orm import joinedload, selectinload
from typing import List, Optional
from sqlalchemy.orm import joinedload, selectinload, sessionmaker
import os, sys, tempfile, textwrap, datetime, signal, subprocess
import pty, fcntl, tty
import time   # <-- add this
import os, sys, shutil, tempfile, textwrap, datetime, time, signal, subprocess
import pty, fcntl, tty
import shutil
from flask import jsonify, request
from flask_login import login_required
from flask_socketio import join_room
# at top of app.py (if not already present)
from flask import render_template, abort
from flask_login import login_required
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import relationship, backref

ROLE_OWNER = "owner"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"


# -------------------------
# Configuration
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(os.path.dirname(__file__), "uploads"))
os.makedirs(STORAGE_DIR, exist_ok=True)

def db_url():
    return os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'bib.db')}")

class Base(DeclarativeBase): pass

class Config:
    # default OFF in production unless explicitly set
    EXEC_SERVER_ENABLED = os.getenv("EXEC_SERVER_ENABLED", "false").lower() in ("1","true","yes","on")
    # optional: only allow specific roles to execute
    EXEC_SERVER_ALLOWED_ROLES = os.getenv("EXEC_SERVER_ALLOWED_ROLES", "owner,admin").split(",")

app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret"),
    SQLALCHEMY_DATABASE_URI=db_url(),
    UPLOAD_DIR=UPLOAD_DIR,
    MAX_CONTENT_LENGTH=1024 * 1024 * 100,  # 100MB
)
app.config.from_object(Config)

# 1) single source of truth
app.config["EXEC_SERVER_ENABLED"] = os.getenv("EXEC_SERVER_ENABLED", "false").lower() in ("1","true","yes","on")
print(">>> EXEC_SERVER_ENABLED =", app.config["EXEC_SERVER_ENABLED"])  # DEBUG: should print True

# 2) make available to *all* templates
@app.context_processor
def inject_flags():
    return {"ENABLE_SERVER_EXEC": app.config["EXEC_SERVER_ENABLED"]}

BLOB_DIR = os.environ.get("BLOB_DIR", os.path.join(app.instance_path, "blobs"))
os.makedirs(BLOB_DIR, exist_ok=True)

ENABLE_SERVER_EXEC = os.environ.get("ENABLE_SERVER_EXEC", "0") == "1"
MAX_EXEC_SECONDS = int(os.environ.get("MAX_EXEC_SECONDS", "8"))
MAX_EXEC_MEMORY_MB = int(os.environ.get("MAX_EXEC_MEMORY_MB", "256"))


engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], future=True, echo=False)
# near the top, after engine is created
SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)


login_manager = LoginManager(app)
login_manager.login_view = "login"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
@app.context_processor
def inject_flags():
    return {"ENABLE_SERVER_EXEC": ENABLE_SERVER_EXEC}


ALLOWED_EXTS = {"pdf", "png", "jpg", "jpeg", "gif", "txt", "md","py","html","mp4","mp3"}

# -------------------------
# Models
# -------------------------
class User(Base, UserMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120), default="")

    memberships: Mapped[List["Membership"]] = relationship(back_populates="user", cascade="all,delete-orphan")
    projects: Mapped[List["Project"]] = relationship(back_populates="owner")

    def set_password(self, pw: str): self.password_hash = generate_password_hash(pw)
    def check_password(self, pw: str) -> bool: return check_password_hash(self.password_hash, pw)

class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    # in Project model
    allow_server_exec: Mapped[bool] = mapped_column(Boolean, default=False)

    owner: Mapped[User] = relationship(back_populates="projects")
    memberships: Mapped[List["Membership"]] = relationship(back_populates="project", cascade="all,delete-orphan")
    pages: Mapped[List["Page"]] = relationship(back_populates="project", cascade="all,delete-orphan")
    citations: Mapped[List["BibEntry"]] = relationship(back_populates="project", cascade="all,delete-orphan")

class Membership(Base):
    __tablename__ = "memberships"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), default="editor")  # owner/editor/viewer

    user: Mapped[User] = relationship(back_populates="memberships")
    project: Mapped[Project] = relationship(back_populates="memberships")
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uix_project_user"),)

class Page(Base):
    __tablename__ = "pages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="Untitled")
    content_html: Mapped[str] = mapped_column(Text, default="")   # condensed UI HTML/notes
    revision: Mapped[int] = mapped_column(Integer, default=0)     # for optimistic concurrency
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    topics: Mapped[List["Topic"]] = relationship(
        back_populates="page", cascade="all,delete-orphan", order_by="Topic.order_index")
    project: Mapped[Project] = relationship(back_populates="pages")
    attachments: Mapped[List["Attachment"]] = relationship(back_populates="page", cascade="all,delete-orphan")
    annotations: Mapped[List["Annotation"]] = relationship(back_populates="page", cascade="all,delete-orphan")
    citations: Mapped[List["BibEntry"]] = relationship(back_populates="page")


class Blob(Base):
    """
    Content-addressed storage (dedup):
      files saved under /uploads/sha256/<first2>/<sha256>[.<ext>]

    - `filename`: original name from the user (kept for download_name)
    - `mime`: canonical MIME type (fallback guessed from filename)
    - `ext`: normalized lower-case file extension without dot (e.g., 'pdf', 'png')
    """

    __tablename__ = "blobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    size: Mapped[int] = mapped_column(Integer)

    # existed before
    mime: Mapped[str] = mapped_column(String(255), default="")

    # NEW: persist extension (pdf, py, png, jpeg, mp4, mp3, …)
    ext: Mapped[Optional[str]] = mapped_column(String(16), default=None)

    # (optional but handy) how many references point to this blob
    refcount: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    # -------- convenience helpers --------
import os, re
from mimetypes import guess_type
from sqlalchemy import select
from flask import send_file, abort

UPLOAD_DIR = os.path.join(app.root_path, "uploads", "sha256")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# what we permit to render inline in the browser
INLINE_MIMES = {
    "application/pdf",
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml",
    "audio/mpeg", "audio/mp4", "audio/ogg", "audio/wav",
    "video/mp4", "video/webm", "video/ogg",
    "text/plain",
}

@app.get("/blob/<string:first2>/<string:sha>/blob")
@login_required
def blob_by_sha(first2, sha):
    # basic validation of sha path
    if len(first2) != 2 or not HEX64.match(sha) or not sha.startswith(first2):
        abort(404)

    # file on disk
    path = os.path.join(UPLOAD_DIR, first2, sha)
    if not os.path.exists(path):
        abort(404)

    # look up metadata (filename + mime) if we have it
    with Session(engine) as s:
        b = s.execute(select(Blob).where(Blob.sha256 == sha)).scalar_one_or_none()

    filename = (b.filename if b and b.filename else sha)
    # 1) prefer stored mime; 2) fallback to guess by filename; 3) octet-stream
    mime = (b.mime if b and b.mime else None) or guess_type(filename)[0] or "application/octet-stream"

    # serve inline only for a safe set, everything else prompts a download
    as_attachment = mime not in INLINE_MIMES

    return send_file(
        path,
        mimetype=mime,
        as_attachment=as_attachment,
        download_name=filename,
        conditional=True,      # supports Range/streaming
        max_age=3600,          # simple cache
        etag=True,
        last_modified=os.path.getmtime(path),
    )

def _save_file_to_storage(file_storage) -> Blob:
    """Accepts a Werkzeug FileStorage, returns a Blob row (creating if new)."""
    raw = file_storage.read()
    sha = hashlib.sha256(raw).hexdigest()

    # Derive safe filename + extension
    original = secure_filename(file_storage.filename or "file")
    base, ext = os.path.splitext(original)
    ext = (ext or "").lower()  # keep leading dot if present, e.g. ".pdf"

    # Best-effort content-type
    ctype = file_storage.mimetype or mimetypes.guess_type(original)[0] or "application/octet-stream"

    # Choose a deterministic on-disk path: uploads/<sha_prefix>/<sha><ext>
    subdir = os.path.join(STORAGE_DIR, sha[:2])
    os.makedirs(subdir, exist_ok=True)
    disk_path = os.path.join(subdir, f"{sha}{ext or ''}")

    # Only write if we don't already have this sha on disk
    if not os.path.exists(disk_path):
        with open(disk_path, "wb") as f:
            f.write(raw)

    size = len(raw)

    # Upsert Blob row
    with SessionLocal() as s:
        blob = s.execute(select(Blob).where(Blob.sha256 == sha)).scalar_one_or_none()
        if blob:
            # If existing record lacked path/ctype/ext/filename, fill them
            if not blob.path:
                blob.path = disk_path
            if not blob.content_type:
                blob.content_type = ctype
            if not blob.ext:
                blob.ext = ext
            if not blob.filename:
                blob.filename = original
            # optional: free DB 'data' if it exists
            if blob.data:
                blob.data = None
        else:
            blob = Blob(
                sha256=sha,
                filename=original,
                ext=ext,
                content_type=ctype,
                size=size,
                path=disk_path,
                data=None  # new rows don't store raw bytes in DB
            )
            s.add(blob)
        s.commit()
        s.refresh(blob)
        return blob

    
    @property
    def storage_basename(self) -> str:
        """Filename used on disk: sha256 or sha256.ext if we know the ext."""
        return f"{self.sha256}.{self.ext}" if self.ext else self.sha256

    @property
    def guessed_mime(self) -> str:
        """MIME, preferring stored `mime`, else guess from filename/ext."""
        if self.mime:
            return self.mime
        # prefer original filename to get richer guesses (e.g., '.md', '.py')
        guess = mimetypes.guess_type(self.filename or "")[0]
        # if that fails, try with ext
        if not guess and self.ext:
            guess = mimetypes.guess_type(f"f.{self.ext}")[0]
        return guess or "application/octet-stream"

    @property
    def download_name(self) -> str:
        """
        A sane default name for downloads, preserving the user's original name.
        Fall back to sha256.ext if `filename` is missing.
        """
        if self.filename:
            return self.filename
        return self.storage_basename

    """
    Content-addressed storage (dedup): files saved under /uploads/sha256/<first2>/<sha256>
    """
    __tablename__ = "blobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    size: Mapped[int] = mapped_column(Integer)
    mime: Mapped[str] = mapped_column(String(255), default="")
    refcount: Mapped[int] = mapped_column(Integer, default=1)
    
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

class Attachment(Base):
    __tablename__ = "attachments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"))
    blob_id: Mapped[int] = mapped_column(ForeignKey("blobs.id"))
    label: Mapped[str] = mapped_column(String(255), default="")

    page: Mapped[Page] = relationship(back_populates="attachments")
    blob: Mapped[Blob] = relationship()

class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"), index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    page: Mapped[Page] = relationship(back_populates="annotations")
    author: Mapped[User] = relationship()

class BibEntry(Base):
    """
    A citation entry; can belong to a Project and optionally be pinned to a Page.
    """
    __tablename__ = "bib_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    page_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pages.id"), nullable=True)
    key: Mapped[str] = mapped_column(String(128), index=True)       # e.g., "smith2020deep"
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[str] = mapped_column(Text, default="")           # "Last, First; Last, First"
    venue: Mapped[str] = mapped_column(String(255), default="")
    year: Mapped[str] = mapped_column(String(8), default="")
    doi: Mapped[str] = mapped_column(String(128), default="")
    url: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    project: Mapped[Project] = relationship(back_populates="citations")
    page: Mapped[Page] = relationship(back_populates="citations")

class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New Topic")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    content_md: Mapped[str] = mapped_column(Text, default="")     # canonical storage
    content_html: Mapped[str] = mapped_column(Text, default="")   # cached render (optional)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    page = relationship("Page", backref=backref("topics", cascade="all,delete-orphan", order_by="Topic.order_index"))
    attachments: Mapped[List["TopicAttachment"]] = relationship(
        back_populates="topic", cascade="all,delete-orphan", lazy="selectin"
    )
    annotations: Mapped[List["TopicAnnotation"]] = relationship(
        back_populates="topic", cascade="all,delete-orphan", lazy="selectin"
    )
    codecells: Mapped[List["TopicCodeCell"]] = relationship(
        back_populates="topic", cascade="all,delete-orphan", lazy="selectin"
    )
    page: Mapped["Page"] = relationship(back_populates="topics")
    attachments: Mapped[List["TopicAttachment"]] = relationship(back_populates="topic", cascade="all,delete-orphan")
    annotations: Mapped[List["TopicAnnotation"]] = relationship(back_populates="topic", cascade="all,delete-orphan")
    codecells: Mapped[List["TopicCodeCell"]] = relationship(back_populates="topic", cascade="all,delete-orphan")


# link Page <-> Topic
Page.topics = relationship("Topic", back_populates="page", cascade="all,delete-orphan", order_by="Topic.order_index")

class TopicAttachment(Base):
    __tablename__ = "topic_attachments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    blob_id: Mapped[int] = mapped_column(ForeignKey("blobs.id"))
    label: Mapped[str] = mapped_column(String(255), default="")
    topic: Mapped[Topic] = relationship(back_populates="attachments")
    blob: Mapped[Blob] = relationship()

class TopicAnnotation(Base):
    __tablename__ = "topic_annotations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    topic: Mapped[Topic] = relationship(back_populates="annotations")
    author: Mapped[User] = relationship()

class TopicCodeCell(Base):
    __tablename__ = "topic_codecells"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    code: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    topic: Mapped[Topic] = relationship(back_populates="codecells")

Base.metadata.create_all(engine)

# -------------------------
# Auth
# -------------------------
@login_manager.user_loader
def load_user(uid):
    with Session(engine) as s: return s.get(User, int(uid))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        name = request.form.get("name", "").strip()
        pw = request.form["password"]
        with Session(engine) as s:
            if s.scalar(select(User).where(User.email == email)):
                flash("Email already registered", "error"); return redirect(url_for("register"))
            u = User(email=email, name=name)
            u.set_password(pw)
            s.add(u); s.commit()
        flash("Registered! Please log in.", "ok")
        return redirect(url_for("login"))
    return render_template("auth_register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        with Session(engine) as s:
            u = s.scalar(select(User).where(User.email == email))
            if not u or not u.check_password(pw):
                flash("Invalid credentials", "error"); return redirect(url_for("login"))
        login_user(u)
        return redirect(url_for("dashboard"))
    return render_template("auth_login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# -------------------------
# Helpers
# -------------------------

def can_edit(contrib):
    """Return True if this contributor has edit privileges."""
    return contrib.role in ("owner", "editor")

def spawn_with_pty(cmd, cwd=None):
    """Start a process attached to a PTY; return (pid, master_fd)."""
    pid, master_fd = pty.fork()
    if pid == 0:
        # Child: exec process
        if cwd:
            os.chdir(cwd)
        # raw mode helps pass through everything
        try:
            tty.setraw(pty.STDIN_FILENO)
        except Exception:
            pass
        os.execvp(cmd[0], cmd)
        os._exit(1)
    # Parent: set non-blocking on master
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return pid, master_fd

def _run_exec_task(tmpdir, room, lang, code, cpu_secs, mem_mb, py_bin):
    runner_py = os.path.join(tmpdir, "runner.py")
    with open(runner_py, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""
            import os, sys
            try:
                import resource
            except Exception:
                resource = None

            CPU_SECS = {cpu_secs}
            MEM_BYTES = {mem_mb} * 1024 * 1024

            def _limit():
                if resource is None:
                    return
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECS, CPU_SECS))
                except Exception:
                    pass
                for name in ("RLIMIT_AS","RLIMIT_DATA","RLIMIT_RSS"):
                    try:
                        lim = getattr(resource, name)
                        resource.setrlimit(lim, (MEM_BYTES, MEM_BYTES))
                        break
                    except Exception:
                        continue
                try:
                    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
                except Exception:
                    pass

            _limit()

            mode = sys.argv[1]
            target = sys.argv[2]
            if mode == "py":
                g = {{}}
                g["__name__"] = "__main__"
                g["__file__"] = target
                with open(target, "rb") as fh:
                    code_obj = compile(fh.read(), target, "exec")
                exec(code_obj, g, g)
            elif mode == "exe":
                target = os.path.abspath(target)
                os.execv(target, [target])
            else:
                print("Unknown mode:", mode)
                sys.exit(2)
        """))

    try:
        # Prepare command (and compile if C++)
        if lang == "python":
            src = os.path.join(tmpdir, "main.py")
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = [py_bin, "-u", runner_py, "py", src]
            preface = "⏳ running Python...\n"
        else:
            compiler = shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                              "stream":"status",
                                              "data":"❌ no C++ compiler found (tried g++, clang++)\n"}, to=room)
                socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": 127}, to=room)
                return
            src = os.path.join(tmpdir, "main.cpp")
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            bin_path = os.path.join(tmpdir, "main_bin")
            compile_cmd = [compiler, "-std=c++17", "-O2", "-pipe", src, "-o", bin_path]

            socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                          "stream":"compiler",
                                          "data":f"🛠️ compiling with {os.path.basename(compiler)}...\n"}, to=room)
            try:
                cp = subprocess.Popen(compile_cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            except Exception as e:
                socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                              "stream":"compiler","data":f"compiler failed to start: {e}\n"}, to=room)
                socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": 1}, to=room)
                return

            # stream compiler
            while True:
                line = cp.stdout.readline() if cp.stdout else ""
                if line:
                    socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                  "stream":"compiler","data":line}, to=room)
                err = cp.stderr.readline() if cp.stderr else ""
                if err:
                    socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                  "stream":"compiler","data":err}, to=room)
                if cp.poll() is not None:
                    if cp.stdout:
                        rem = cp.stdout.read()
                        if rem: socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                              "stream":"compiler","data":rem}, to=room)
                    if cp.stderr:
                        rem = cp.stderr.read()
                        if rem: socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                              "stream":"compiler","data":rem}, to=room)
                    break
            if cp.returncode != 0:
                socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": cp.returncode}, to=room)
                return

            cmd = [py_bin, "-u", runner_py, "exe", bin_path]
            preface = "🚀 running C++...\n"

        # Run under PTY and stream raw chunks
        try:
            pid, master = spawn_with_pty(cmd, cwd=tmpdir)
        except Exception as e:
            socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                          "stream":"status","data":f"failed to start: {e}\n"}, to=room)
            socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": 1}, to=room)
            return

        socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                      "stream":"status","data":preface}, to=room)

        deadline = datetime.datetime.utcnow() + datetime.timedelta(seconds=cpu_secs * 2)
        while True:
            if datetime.datetime.utcnow() > deadline:
                try: os.kill(pid, signal.SIGKILL)
                except Exception: pass
                socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                              "stream":"status","data":"⛔ timed out\n"}, to=room)
                socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": 124}, to=room)
                break
            try:
                chunk = os.read(master, 4096)
                if chunk:
                    socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                  "stream":"stdout","data":chunk.decode(errors="replace")}, to=room)
            except BlockingIOError:
                pass
            # exit check
            try:
                ended, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                ended, status = pid, 0
            if ended == pid:
                # drain
                try:
                    while True:
                        chunk = os.read(master, 4096)
                        if not chunk: break
                        socketio.emit("code_output", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0),
                                                      "stream":"stdout","data":chunk.decode(errors="replace")}, to=room)
                except Exception:
                    pass
                rc = os.waitstatus_to_exitcode(status) if 'status' in locals() else 0
                socketio.emit("code_done", {"topic_id": int(os.path.basename(tmpdir).split("_")[-1] or 0), "returncode": rc}, to=room)
                break
            time.sleep(0.05)
        try:
            os.close(master)
        except Exception:
            pass
    finally:
        # Clean up temp dir after we finish streaming
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

def _as_bool(v, default=False):
    if isinstance(v, bool): return v
    if not v: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def require_project_editor(project_id: int):
    if not current_user.is_authenticated:
        abort(403)
    with SessionLocal() as s:
        proj = s.get(Project, project_id)
        if not proj: abort(404)
        if proj.owner_id == current_user.id:
            return True
        m = s.scalar(select(Membership).where(
            Membership.project_id == project_id,
            Membership.user_id == current_user.id
        ))
        if not m: abort(403)
        if m.role not in ("editor",):
            abort(403)
        return True

def run_code_sandbox(code: str, timeout_sec: int = 2):
    """
    Executes Python in a restricted subprocess with CPU/memory caps.
    NOTE: This is a minimal sandbox for local/development. For production, isolate
    via containers, seccomp, or dedicated workers.
    """
    import sys, subprocess, tempfile, os, signal
    try:
        import resource  # POSIX only
    except Exception:
        resource = None

    def preexec():
        if resource:
            # 2s CPU, 256MB address space
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            try:
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            except Exception:
                pass
        os.setsid()

    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONSAFEPATH": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                preexec_fn=preexec if hasattr(os, "setsid") else None,
                env=env,
            )
            return {"ok": True, "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "error": "timeout", "stdout": e.stdout or "", "stderr": e.stderr or ""}
        except Exception as e:
            return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

import hashlib, os
from mimetypes import guess_type
from sqlalchemy import select

UPLOAD_DIR = os.path.join(app.root_path, "uploads", "sha256")

def stash_blob(fs, *, filename=None, mime=None) -> int:
    """Save file bytes once (by sha256), persist Blob metadata, and return blob_id."""
    # read the stream once
    data = fs.read()
    sha256 = hashlib.sha256(data).hexdigest()
    rel = os.path.join(sha256[:2], sha256)
    path = os.path.join(UPLOAD_DIR, rel)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)

    # normalize metadata
    orig_name = filename or fs.filename or "file"
    safe_name = secure_filename(orig_name) or "file"
    guessed = guess_type(safe_name)[0]  # e.g. 'application/pdf'
    mime_final = mime or fs.mimetype or guessed or ""

    with Session(engine) as s:
        existing = s.execute(select(Blob).where(Blob.sha256 == sha256)).scalar_one_or_none()
        if existing:
            # optional: update empty metadata if we learned better info
            if not existing.filename:
                existing.filename = safe_name
            if (not existing.mime) and mime_final:
                existing.mime = mime_final
            # optional: bump refcount
            if hasattr(existing, "refcount"):
                existing.refcount = (existing.refcount or 0) + 1
            s.commit()
            return existing.id

        b = Blob(
            sha256=sha256,
            filename=safe_name,
            size=len(data),
            mime=mime_final,
            refcount=1 if hasattr(Blob, "refcount") else None,
        )
        s.add(b)
        s.flush()   # assigns b.id without closing session
        blob_id = b.id
        s.commit()
        return blob_id

    data = file.read()
    sha256 = hashlib.sha256(data).hexdigest()
    path = os.path.join(UPLOAD_DIR, sha256[:2], sha256)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)

    with Session(engine) as s:
        blob = s.query(Blob).filter_by(sha256=sha256).first()
        if not blob:
            blob = Blob(
                sha256=sha256,
                filename=filename or file.filename,
                size=len(data),
                mime=mime or file.mimetype or "",
            )
            s.add(blob)
            s.commit()
        return blob

    data = file_storage.read()
    sha = hashlib.sha256(data).hexdigest()
    first2 = sha[:2]
    folder = os.path.join(app.config["UPLOAD_DIR"], first2, sha)
    path = os.path.join(folder, "blob")
    os.makedirs(folder, exist_ok=True)

    # If not present, write it once
    if not os.path.exists(path):
        with open(path, "wb") as f: f.write(data)

    # Upsert Blob row
    with Session(engine) as s:
        b = s.scalar(select(Blob).where(Blob.sha256 == sha))
        if b:
            b.refcount += 1
        else:
            b = Blob(
                sha256=sha,
                filename=secure_filename(file_storage.filename or "file"),
                size=len(data),
                mime=file_storage.mimetype or "",
                refcount=1,
            )
            s.add(b)
        s.commit()
        return b.id

def require_project_member(project_id: int):
    if not current_user.is_authenticated: abort(403)
    with Session(engine) as s:
        proj = s.get(Project, project_id)
        if not proj: abort(404)
        if proj.owner_id == current_user.id: return proj
        m = s.scalar(select(Membership).where(Membership.project_id == project_id, Membership.user_id == current_user.id))
        if not m: abort(403)
        return proj

def require_project_owner(project_id: int):
    if not current_user.is_authenticated:
        abort(403)
    with SessionLocal() as s:
        proj = s.get(Project, project_id)
        if not proj:
            abort(404)
        if proj.owner_id != current_user.id:
            abort(403)
        return True


# -------------------------
# Views
# -------------------------
@app.route("/project/<int:project_id>/edit", methods=["POST"])
@login_required
def project_edit(project_id):
    require_project_editor(project_id)
    project = db.session.get(Project, project_id)
    contrib = project.get_contrib(current_user.id)
    if not contrib or not can_edit(contrib):
        abort(403, "You do not have permission to edit this project.")

    # proceed with edit...
    project.title = request.form["title"]
    db.session.commit()
    return redirect(url_for("project_view", project_id=project.id))

@app.route("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/edit", methods=["POST"])
@login_required
def topic_edit(project_id, page_id, topic_id):
    require_project_editor(project_id)
    topic = db.session.get(Topic, topic_id)
    contrib = topic.page.project.get_contrib(current_user.id)
    if not contrib or not can_edit(contrib):
        abort(403)

    topic.content_md = request.form["content"]
    db.session.commit()
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic.id))

@app.get("/blob/<int:blob_id>")
@login_required
def get_blob(blob_id):
    with Session(engine) as s:
        b = s.get(Blob, blob_id)
        if not b:
            abort(404)
        path = os.path.join(UPLOAD_DIR, b.sha256[:2], b.sha256)

    return send_file(
        path,
        mimetype=b.mime or "application/octet-stream",
        as_attachment=True,               # or False if you want inline
        download_name=b.filename or "file"
    )

@app.post("/project/<int:project_id>/page/<int:page_id>/attach")
@login_required
def page_attach(project_id, page_id):
    require_project_editor(project_id)
    file = request.files.get("file")
    if not file:
        return redirect(url_for("page_view", project_id=project_id, page_id=page_id))

    blob = _save_file_to_storage(file)   # <-- NEW helper

    att = Attachment(
        page_id=page_id,
        blob_id=blob.id,
        label=request.form.get("label") or blob.filename
    )
    with SessionLocal() as s:
        s.add(att)
        s.commit()

    return redirect(url_for("page_view", project_id=project_id, page_id=page_id))


@app.get("/files/<sha_prefix>/<sha>/<filename>")
def serve_file(sha_prefix: str, sha: str, filename: str):
    """Serve stored files with correct mimetype and filename."""
    if sha_prefix != sha[:2]:
        abort(404)

    with SessionLocal() as s:
        blob = s.execute(select(Blob).where(Blob.sha256 == sha)).scalar_one_or_none()
        if not blob:
            abort(404)

    # Prefer filesystem path
    if blob.path and os.path.exists(blob.path):
        # Infer mimetype if missing
        mime = blob.content_type or mimetypes.guess_type(blob.filename)[0] or "application/octet-stream"
        # Let browsers render PDFs/images/video inline; support range/conditional
        return send_file(
            blob.path,
            mimetype=mime,
            as_attachment=False,
            download_name=blob.filename,  # sets Content-Disposition with filename
            conditional=True,             # enables If-Modified-Since/ETag
            max_age=3600,
            etag=True
        )

    # Legacy fallback: stored in DB as bytes
    if blob.data:
        mime = blob.content_type or mimetypes.guess_type(blob.filename)[0] or "application/octet-stream"
        resp = Response(blob.data, mimetype=mime, direct_passthrough=True)
        # Present as inline with the original name
        resp.headers["Content-Disposition"] = f'inline; filename="{blob.filename}"'
        return resp

    abort(404)

@app.post("/projects/<int:project_id>/pages/<int:page_id>/topics/<int:topic_id>/export/html", endpoint="topic_export_html")
@login_required
def topic_export_html(project_id, page_id, topic_id):
    title = request.form.get("title") or "document"
    html  = request.form.get("content_html") or "<!doctype html><title>Empty</title><body>Empty</body>"
    from datetime import datetime
    fname = f"{secure_filename(title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return Response(
        html,
        mimetype="text/html",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )

# PDF export: adapt to your generator (WeasyPrint, wkhtmltopdf, xhtml2pdf, FPDF+html2text, etc.)
@app.post("/projects/<int:project_id>/pages/<int:page_id>/topics/<int:topic_id>/export/pdf", endpoint="topic_export_pdf")
@login_required
def topic_export_pdf(project_id, page_id, topic_id):
    title = request.form.get("title") or "document"
    html  = request.form.get("content_html") or "<!doctype html><title>Empty</title><body>Empty</body>"

    # Example with WeasyPrint (best for CSS, SVG MathJax):
    # from weasyprint import HTML
    # pdf_bytes = HTML(string=html, base_url=request.host_url).write_pdf()

    # If you’re using your current FPDF pipeline, convert HTML→text or use an HTML-capable lib.

    pdf_bytes = b"%PDF-1.4\n% ... replace with real PDF bytes ..."
    from datetime import datetime
    fname = f"{secure_filename(title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )
@app.post("/project/<int:project_id>/page/<int:page_id>/delete")
@login_required
def page_delete(project_id, page_id):
    # require_project_owner_or_editor(project_id)

    with SessionLocal() as s:
        page = s.get(Page, page_id)
        if not page or page.project_id != project_id:
            abort(404)

        # If you defined relationships with cascade="all, delete-orphan"
        # this is enough:
        s.delete(page)
        s.commit()

    flash("Page deleted", "success")
    return redirect(url_for("project_view", project_id=project_id))

# @app.post("/project/<int:project_id>/page/<int:page_id>/delete")
# @login_required
# def page_delete(project_id: int, page_id: int):
#     require_project_editor(project_id)

#     with SessionLocal() as s:
#         page = s.execute(
#             select(Page)
#             .where(Page.id == page_id, Page.project_id == project_id)
#             .options(
#                 selectinload(Page.topics).selectinload(Topic.codecells),
#                 selectinload(Page.attachments),
#                 selectinload(Page.annotations)
#             )
#         ).scalar_one_or_none()
#         if not page:
#             abort(404)

#         topic_ids = [t.id for t in page.topics]
#         cell_ids  = [c.id for t in page.topics for c in t.codecells]

#         # Delete children first
#         if cell_ids:
#             s.execute(delete(TopicCodeCell).where(TopicCodeCell.id.in_(cell_ids)))
#         if topic_ids:
#             s.execute(delete(Topic).where(Topic.id.in_(topic_ids)))

#         s.execute(delete(Attachment).where(Attachment.page_id == page_id))
#         s.execute(delete(Annotation).where(Annotation.page_id == page_id))
#         # s.execute(delete(Citation).where(Citation.page_id == page_id))

#         # Delete page
#         s.execute(delete(Page).where(Page.id == page_id))
#         s.commit()

#     try:
#         socketio.emit("page_deleted", {"project_id": project_id, "page_id": page_id}, to=f"project:{project_id}")
#     except Exception:
#         pass

#     flash("Page deleted.", "success")
#     return redirect(url_for("project_view", project_id=project_id))

from sqlalchemy import select, update, delete, func

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/delete")
@login_required
def topic_delete(project_id: int, page_id: int, topic_id: int):
    require_project_editor(project_id)

    with SessionLocal() as s:
        topic = s.execute(
            select(Topic)
            .where(Topic.id == topic_id, Topic.page_id == page_id)
            .options(
                selectinload(Topic.codecells),
                selectinload(Topic.attachments),      # if you store topic-level attachments
                selectinload(Topic.annotations)       # if you store topic-level annotations
            )
        ).scalar_one_or_none()
        if not topic:
            abort(404)

        cell_ids = [c.id for c in topic.codecells]

        if cell_ids:
            s.execute(delete(TopicCodeCell).where(TopicCodeCell.id.in_(cell_ids)))

        # If your schema has topic-level attachments/annotations:
        s.execute(delete(Attachment).where(Attachment.topic_id == topic_id))
        s.execute(delete(Annotation).where(Annotation.topic_id == topic_id))

        # If citations can be scoped at topic-level:
        s.execute(delete(Citation).where(Citation.topic_id == topic_id))

        # Finally delete topic
        s.execute(delete(Topic).where(Topic.id == topic_id))
        s.commit()

    try:
        socketio.emit("topic_deleted", {
            "project_id": project_id, "page_id": page_id, "topic_id": topic_id
        }, to=f"project:{project_id}")
    except Exception:
        pass

    flash("Topic deleted.", "success")
    return redirect(url_for("page_view", project_id=project_id, page_id=page_id))

@app.route("/")
@login_required
def dashboard():
    with Session(engine) as s:
        my_projects = s.scalars(select(Project).where(Project.owner_id == current_user.id)).all()
        shared = s.scalars(
            select(Project).join(Membership).where(Membership.user_id == current_user.id)
        ).all()
    return render_template("dashboard.html", my_projects=my_projects, shared_projects=shared)

@app.post("/project/<int:project_id>/page/<int:page_id>/rename")
@login_required
def page_rename(project_id, page_id):
    # Any project member can rename pages; change to require_project_owner if you want owner-only
    require_project_editor(project_id)
    new_title = (request.form.get("title") or "").strip()
    if not new_title:
        flash("Title cannot be empty.", "error")
        return redirect(url_for("page_view", project_id=project_id, page_id=page_id))

    with SessionLocal() as s:
        page = s.execute(
            select(Page).where(Page.id == page_id, Page.project_id == project_id)
        ).scalar_one_or_none()
        if not page:
            abort(404)
        page.title = new_title
        s.commit()
        pid = page.id

    # notify collaborators viewing this page
    socketio.emit(
        "page_title_updated",
        {"page_id": page_id, "title": new_title},
        to=f"page:{page_id}"
    )
    return redirect(url_for("page_view", project_id=project_id, page_id=pid))

@app.post("/project/<int:project_id>/members/<int:member_id>/role")
@login_required
def member_change_role(project_id, member_id):
    require_project_owner(project_id)
    role = (request.form.get("role") or "").strip().lower()
    if role not in {"editor", "viewer"}:
        flash("Invalid role.", "error")
        return redirect(url_for("project_view", project_id=project_id))
    with SessionLocal() as s:
        m = s.get(Membership, member_id)
        if not m or m.project_id != project_id:
            abort(404)
        # don't allow changing owner via membership rows
        if m.user_id == current_user.id:
            flash("You are the owner; change not applicable here.", "error")
            return redirect(url_for("project_view", project_id=project_id))
        m.role = role
        s.commit()
        uid = m.user_id
    socketio.emit("member_updated", {"project_id": project_id, "member_id": member_id, "role": role}, to=f"project:{project_id}")
    flash("Role updated.", "ok")
    return redirect(url_for("project_view", project_id=project_id))


@app.post("/project/<int:project_id>/members/<int:member_id>/remove")
@login_required
def member_remove(project_id, member_id):
    require_project_owner(project_id)
    with SessionLocal() as s:
        m = s.get(Membership, member_id)
        if not m or m.project_id != project_id:
            abort(404)
        # prevent removing self/owner via memberships
        if m.user_id == current_user.id:
            flash("Owner cannot be removed.", "error")
            return redirect(url_for("project_view", project_id=project_id))
        s.delete(m)
        s.commit()
    socketio.emit("member_removed", {"project_id": project_id, "member_id": member_id}, to=f"project:{project_id}")
    flash("Member removed.", "ok")
    return redirect(url_for("project_view", project_id=project_id))

@app.route("/project/create", methods=["POST"])
@login_required
def project_create():
    name = request.form["name"].strip()
    if not name:
        flash("Project name required", "error")
        return redirect(url_for("dashboard"))
    with Session(engine) as s:
        p = Project(name=name, owner_id=current_user.id)
        s.add(p)
        s.flush()               # ensures p.id is assigned by the DB
        pid = p.id              # capture while bound
        s.commit()
    return redirect(url_for("project_view", project_id=pid))

@app.route("/project/<int:project_id>")
@login_required
def project_view(project_id):
    # keep permission check, but don't pass this detached object to Jinja
    require_project_member(project_id)

    with SessionLocal() as s:
        proj = s.execute(
            select(Project)
            .where(Project.id == project_id)
            .options(
                joinedload(Project.owner),          # eager load owner (many-to-one)
            )
        ).scalar_one_or_none()
        if not proj:
            abort(404)

        pages = s.scalars(
            select(Page)
            .where(Page.project_id == project_id)
            .order_by(Page.updated_at.desc())
        ).all()

        members = s.execute(
            select(Membership)
            .where(Membership.project_id == project_id)
            .options(selectinload(Membership.user))  # eager load user on membership
        ).scalars().all()

        cits = s.scalars(
            select(BibEntry)
            .where(BibEntry.project_id == project_id)
            .order_by(BibEntry.updated_at.desc())
        ).all()

    return render_template("project.html", project=proj, pages=pages, members=members, citations=cits)


@app.route("/project/<int:project_id>/invite", methods=["POST"])
@login_required
def project_invite(project_id):
    # proj = require_project_member(project_id)
    require_project_editor(project_id=project_id)
    email = request.form["email"].strip().lower()
    with Session(engine) as s:
        u = s.scalar(select(User).where(User.email == email))
        if not u:
            flash("User not found. Ask them to register first.", "error")
            return redirect(url_for("project_view", project_id=project_id))
        existing = s.scalar(select(Membership).where(Membership.project_id == project_id, Membership.user_id == u.id))
        if existing:
            flash("Already a member.", "ok")
        else:
            m = Membership(project_id=project_id, user_id=u.id, role="editor")
            s.add(m); s.commit()
    # realtime notice
    socketio.emit("member_added", {"project_id": project_id, "email": email}, to=f"project:{project_id}")
    return redirect(url_for("project_view", project_id=project_id))

@app.route("/project/<int:project_id>/page/create", methods=["POST"])
@login_required
def page_create(project_id):
    proj = require_project_editor(project_id)
    title = request.form.get("title", "Untitled").strip() or "Untitled"
    with Session(engine) as s:
        page = Page(project_id=project_id, title=title, content_html="")
        s.add(page); s.commit()
        pid = page.id
    return redirect(url_for("page_view", project_id=project_id, page_id=pid))

@app.route("/project/<int:project_id>/page/<int:page_id>")
@login_required
def page_view(project_id, page_id):
    # permission check (doesn't return ORM to template)
    require_project_member(project_id)

    with SessionLocal() as s:
        # Load the project (owner used in base/project nav sometimes)
        proj = s.execute(
            select(Project)
            .where(Project.id == project_id)
            .options(joinedload(Project.owner))
        ).scalar_one_or_none()
        if not proj:
            abort(404)

        # Eager-load page and all relationships referenced in the template:
        # - attachments -> blob
        # - annotations -> author
        # - citations (page-level)
        page = s.execute(
        select(Page)
        .where(Page.id == page_id, Page.project_id == project_id)
        .options(
            selectinload(Page.attachments).selectinload(Attachment.blob),
            selectinload(Page.annotations).selectinload(Annotation.author),
            selectinload(Page.citations),
            selectinload(Page.topics),   # <-- add this
        )
        ).scalar_one_or_none()

        if not page:
            abort(404)

        # Since everything is eager-loaded, it’s safe to grab lists now
        atts = list(page.attachments)
        anns = list(page.annotations)
        cits = list(page.citations)

    return render_template(
        "page.html",
        project=proj,
        page=page,
        attachments=atts,
        annotations=anns,
        citations=cits,
    )


from werkzeug.utils import secure_filename

@app.post("/project/<int:project_id>/page/<int:page_id>/upload")
@login_required
def upload(project_id, page_id):
    require_project_editor(project_id)
    file = request.files.get("file")
    label = (request.form.get("label") or "").strip()

    if not file or not allowed_file(file.filename):
        flash("Unsupported or missing file", "error")
        return redirect(url_for("page_view", project_id=project_id, page_id=page_id))

    blob_id = stash_blob(file, filename=file.filename, mime=file.mimetype)

    with Session(engine) as s:
        a = Attachment(
            page_id=page_id,
            blob_id=blob_id,
            label=label or secure_filename(file.filename) or "file"
        )
        s.add(a)
        s.commit()

    socketio.emit("file_added", {"page_id": page_id}, to=f"page:{page_id}")
    return redirect(url_for("page_view", project_id=project_id, page_id=page_id))



@app.get("/blob/<sha_prefix>/<sha>/blob")
def serve_blob(sha_prefix, sha):
    # Best served behind auth + signed URLs in production
    folder = os.path.join(app.config["UPLOAD_DIR"], sha_prefix, sha)
    path = os.path.join(folder, "blob")
    if not os.path.exists(path): abort(404)
    # Content-disposition left inline so PDF/images preview in browser
    return send_from_directory(folder, "blob")

@app.post("/project/<int:project_id>/page/<int:page_id>/annotate")
@login_required
def annotate(project_id, page_id):
    proj = require_project_editor(project_id)
    
    text = request.form.get("text", "").strip()
    if not text: return redirect(url_for("page_view", project_id=project_id, page_id=page_id))
    with Session(engine) as s:
        an = Annotation(page_id=page_id, author_id=current_user.id, text=text)
        s.add(an); s.commit()
    socketio.emit("annotation_added", {"page_id": page_id, "author": current_user.email, "text": text}, to=f"page:{page_id}")
    return redirect(url_for("page_view", project_id=project_id, page_id=page_id))

@app.post("/project/<int:project_id>/page/<int:page_id>/save_html")
@login_required
def save_html(project_id, page_id):
    # Used by Socket events or fallback POST to persist the condensed UI content
    proj = require_project_editor(project_id)
    content_html = request.form.get("content_html", "")
    client_rev = int(request.form.get("revision", "0"))
    with Session(engine) as s:
        page = s.get(Page, page_id)
        if not page or page.project_id != project_id: abort(404)
        if client_rev != page.revision:
            return jsonify({"status": "conflict", "server_revision": page.revision, "server_html": page.content_html}), 409
        page.content_html = content_html
        page.revision += 1
        s.commit()
        new_rev = page.revision
    socketio.emit("page_saved", {"page_id": page_id, "revision": new_rev}, to=f"page:{page_id}")
    return jsonify({"status": "ok", "revision": new_rev})

# Create a topic under a page
@app.post("/project/<int:project_id>/page/<int:page_id>/topic/create")
@login_required
def topic_create(project_id, page_id):
    require_project_editor(project_id)
    title = (request.form.get("title") or "New topic").strip()
    with SessionLocal() as s:
        page = s.scalar(select(Page).where(Page.id == page_id, Page.project_id == project_id))
        if not page: abort(404)
        # next order index
        max_idx = s.scalar(select(func.coalesce(func.max(Topic.order_index), -1)).where(Topic.page_id == page_id)) or -1
        t = Topic(page_id=page_id, title=title, order_index=max_idx + 1)
        s.add(t); s.commit()
        tid = t.id
    socketio.emit("topic_added", {"page_id": page_id, "topic_id": tid}, to=f"page:{page_id}")
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=tid))


@app.get("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>")
@login_required
def topic_view(project_id, page_id, topic_id):
    require_project_member(project_id)
    with SessionLocal() as s:
        proj = s.execute(
            select(Project)
            .where(Project.id == project_id)
            .options(joinedload(Project.owner))
        ).scalar_one_or_none()
        if not proj: abort(404)

        page = s.execute(
            select(Page).where(Page.id == page_id, Page.project_id == project_id)
        ).scalar_one_or_none()
        if not page: abort(404)

        topic = s.execute(
            select(Topic)
            .where(Topic.id == topic_id, Topic.page_id == page_id)
            .options(
                selectinload(Topic.attachments).selectinload(TopicAttachment.blob),
                selectinload(Topic.annotations).selectinload(TopicAnnotation.author),
                selectinload(Topic.codecells),
            )
        ).scalar_one_or_none()
        if not topic: abort(404)

        atts = list(topic.attachments)
        anns = list(topic.annotations)
        cells = list(topic.codecells)

    return render_template(
        "topic.html",
        project=proj,
        page=page,
        topic=topic,
        attachments=atts,
        annotations=anns,
        codecells=cells,
    )

# Save topic markdown (optimistic concurrency)

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/save")
@login_required
def topic_save(project_id, page_id, topic_id):
    # require_project_member(project_id)
    require_project_editor(project_id=project_id)
    md = request.form.get("content_md", "")
    client_rev = int(request.form.get("revision", "0"))
    with SessionLocal() as s:
        t = s.get(Topic, topic_id)
        if not t or t.page_id != page_id: abort(404)
        if client_rev != t.revision:
            return jsonify({"status": "conflict", "server_revision": t.revision, "server_md": t.content_md}), 409
        t.content_md = md
        t.revision += 1
        s.commit()
        rev = t.revision
    socketio.emit("topic_saved", {"topic_id": topic_id, "revision": rev}, to=f"topic:{topic_id}")
    return jsonify({"status": "ok", "revision": rev})

# Upload attachment to a topic
@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/upload")
@login_required
def topic_upload(project_id, page_id, topic_id):
    require_project_editor(project_id)
    file = request.files.get("file")
    label = (request.form.get("label") or "").strip()
    if not file or not allowed_file(file.filename):
        flash("Unsupported or missing file", "error")
        return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))
    blob_id = stash_blob(file)
    with SessionLocal() as s:
        t = s.get(Topic, topic_id)
        if not t or t.page_id != page_id: abort(404)
        ta = TopicAttachment(topic_id=topic_id, blob_id=blob_id, label=label)
        s.add(ta); s.commit()
    socketio.emit("topic_file_added", {"topic_id": topic_id}, to=f"topic:{topic_id}")
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))

# Drag/drop reorder topics on a page
@app.post("/project/<int:project_id>/page/<int:page_id>/topic/reorder")
@login_required
def topic_reorder(project_id, page_id):
    require_project_member(project_id)
    order = request.json.get("order", [])  # list of topic IDs in new order
    with SessionLocal() as s:
        topics = s.scalars(select(Topic).where(Topic.page_id == page_id)).all()
        idx_map = {tid: i for i, tid in enumerate(order)}
        for t in topics:
            if t.id in idx_map:
                t.order_index = idx_map[t.id]
        s.commit()
    socketio.emit("topics_reordered", {"page_id": page_id, "order": order}, to=f"page:{page_id}")
    return jsonify({"status": "ok"})

# Execute Python code in a topic
@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/run")
@login_required
def topic_run(project_id, page_id, topic_id):
    require_project_member(project_id)
    code = request.form.get("code", "")
    result = run_code_sandbox(code, timeout_sec=3)
    return jsonify(result)


@app.post("/project/<int:project_id>/page/<int:page_id>/topics/reorder")
@login_required
def topics_reorder(project_id, page_id):
    require_project_member(project_id)
    ids = request.json.get("order", [])
    if not isinstance(ids, list):
        return jsonify({"error": "bad payload"}), 400
    with SessionLocal() as s:
        cur = s.scalars(select(Topic).where(Topic.page_id == page_id, Topic.id.in_(ids))).all()
        # map id -> topic
        m = {t.id: t for t in cur}
        for idx, tid in enumerate(ids):
            tt = m.get(int(tid))
            if tt:
                tt.order_index = idx
        s.commit()
    socketio.emit("topics_reordered", {"page_id": page_id, "order": ids}, to=f"page:{page_id}")
    return jsonify({"status":"ok"})

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/save_md")
@login_required
def topic_save_md(project_id, page_id, topic_id):
    require_project_editor(project_id)
    md = request.form.get("content_md", "")
    client_rev = int(request.form.get("revision", "0"))
    with SessionLocal() as s:
        t = s.get(Topic, topic_id)
        if not t or t.page_id != page_id: abort(404)
        if client_rev != t.revision:
            return jsonify({"status":"conflict","server_revision":t.revision,"server_md":t.content_md}), 409
        t.content_md = md
        t.revision += 1
        s.commit()
        new_rev = t.revision
    socketio.emit("topic_broadcast", {"topic_id": topic_id, "md": md, "revision": new_rev}, to=f"topic:{topic_id}")
    return jsonify({"status":"ok","revision":new_rev})


@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/annotate")
@login_required
def topic_annotate(project_id, page_id, topic_id):
    require_project_editor(project_id)
    text = (request.form.get("text") or "").strip()
    if not text:
        return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))
    with SessionLocal() as s:
        an = TopicAnnotation(topic_id=topic_id, author_id=current_user.id, text=text)
        s.add(an); s.commit()
    socketio.emit("topic_annotation_added", {"topic_id": topic_id}, to=f"topic:{topic_id}")
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/codecell/create")
@login_required
def codecell_create(project_id, page_id, topic_id):
    # require_project_member(project_id)
    require_project_editor(project_id=project_id)
    with SessionLocal() as s:
        max_idx = s.scalar(select(func.coalesce(func.max(TopicCodeCell.order_index), -1)).where(TopicCodeCell.topic_id == topic_id))
        cc = TopicCodeCell(topic_id=topic_id, order_index=(max_idx + 1), code="")
        s.add(cc); s.commit()
        cid = cc.id
    socketio.emit("codecell_added", {"topic_id": topic_id, "cell_id": cid}, to=f"topic:{topic_id}")
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/codecell/<int:cell_id>/save")
@login_required
def codecell_save(project_id, page_id, topic_id, cell_id):
    # require_project_member(project_id)
    require_project_editor(project_id)
    code = request.form.get("code", "")
    with SessionLocal() as s:
        cc = s.get(TopicCodeCell, cell_id)
        if not cc or cc.topic_id != topic_id: abort(404)
        cc.code = code
        s.commit()
    socketio.emit("codecell_updated", {"topic_id": topic_id, "cell_id": cell_id}, to=f"topic:{topic_id}")
    return jsonify({"status":"ok"})

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/codecell/<int:cell_id>/delete")
@login_required
def codecell_delete(project_id, page_id, topic_id, cell_id):
    require_project_editor(project_id)
    with SessionLocal() as s:
        cc = s.get(TopicCodeCell, cell_id)
        if not cc or cc.topic_id != topic_id: abort(404)
        s.delete(cc); s.commit()
    socketio.emit("codecell_deleted", {"topic_id": topic_id, "cell_id": cell_id}, to=f"topic:{topic_id}")
    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))

import sys, tempfile, textwrap, selectors, signal, resource, shlex, json, subprocess

def _limit_resources():
    # CPU time (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_EXEC_SECONDS, MAX_EXEC_SECONDS))
    # Max address space (bytes)
    rss = MAX_EXEC_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (rss, rss))
    # No core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import sys, tempfile, selectors, datetime, subprocess, os, textwrap, json

# --- Route: Python & C++ execution with resource-limited runner and PTY streaming
def _exec_is_allowed():
    if not current_app.config["EXEC_SERVER_ENABLED"]:
        return False, "[server] Disabled. Ask project owner to enable server execution."
    # optional: role-based control
    allowed = [r.strip().lower() for r in current_app.config["EXEC_SERVER_ALLOWED_ROLES"]]
    # adapt this to your user model (e.g., current_user.role or is_admin flag)
    user_role = (getattr(current_user, "role", "user") or "user").lower()
    if allowed and user_role not in allowed:
        return False, "[server] Disabled for your role."
    return True, None

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/exec")
@login_required
def exec_server_code(project_id, page_id, topic_id):
    if not current_app.config["EXEC_SERVER_ENABLED"]:
        return jsonify({"ok": False, "error": "[server] Disabled. Ask project owner to enable server execution."}), 403
    ok, msg = _exec_is_allowed()
    if not ok:
        return jsonify({"ok": False, "error": msg}), 403
    if not ENABLE_SERVER_EXEC:
        return jsonify({"error": "server execution disabled"}), 403
    require_project_editor(project_id)

    payload = request.get_json(silent=True) or {}
    code = str(payload.get("code") or "")
    lang = (payload.get("lang") or "python").lower()
    if not code.strip():
        return jsonify({"error": "empty code"}), 400
    if lang not in ("python", "cpp"):
        return jsonify({"error": "unsupported language"}), 400

    tmpdir = tempfile.mkdtemp(prefix=f"topic_{topic_id}_")
    room = f"topic:{topic_id}"
    py_bin = sys.executable
    cpu_secs = MAX_EXEC_SECONDS
    mem_mb = MAX_EXEC_MEMORY_MB

    # fire-and-forget background task that will emit output to sockets
    socketio.start_background_task(_run_exec_task, tmpdir, room, lang, code, cpu_secs, mem_mb, py_bin)

    # immediate OK so the client can show "Started. Streaming…"
    return jsonify({"status": "started"})


@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/rename")
@login_required
def topic_rename(project_id: int, page_id: int, topic_id: int):
    """
    Rename a topic. Accepts form-encoded or JSON:
      - form:    title=<new title>
      - json:    {"title": "..."}
    Returns JSON if the request is AJAX/JSON; otherwise redirects back to the topic page.
    """
    require_project_editor(project_id)

    # get payload
    new_title = (request.form.get("title")
                 or (request.is_json and (request.get_json(silent=True) or {}).get("title"))
                 or "").strip()

    if not new_title:
        # for JSON callers
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "title is required"}), 400
        # for form callers
        return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))

    with SessionLocal() as s:
        topic = s.execute(
            select(Topic).where(Topic.id == topic_id, Topic.page_id == page_id)
        ).scalar_one_or_none()
        if not topic:
            abort(404)

        old_title = topic.title
        topic.title = new_title
        s.commit()

    # notify collaborators in the project room (optional)
    try:
        socketio.emit(
            "topic_renamed",
            {"project_id": project_id, "page_id": page_id, "topic_id": topic_id,
             "old_title": old_title, "new_title": new_title},
            to=f"project:{project_id}",
        )
    except Exception:
        pass

    # JSON for AJAX; redirect for regular form
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"ok": True, "title": new_title})

    return redirect(url_for("topic_view", project_id=project_id, page_id=page_id, topic_id=topic_id))


@app.post("/project/<int:project_id>/toggle_server_exec")
@login_required
def toggle_server_exec(project_id):
    require_project_owner(project_id)
    with SessionLocal() as s:
        p = s.get(Project, project_id); 
        if not p: abort(404)
        p.allow_server_exec = not p.allow_server_exec
        s.commit()
    flash(f"Server exec is now {'ON' if p.allow_server_exec else 'OFF'} for this project","ok")
    return redirect(url_for('project_view', project_id=project_id))


@app.get("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/code")
@login_required
def topic_code_workspace(project_id, page_id, topic_id):
    require_project_member(project_id)
    with SessionLocal() as s:
        proj = s.execute(
            select(Project).where(Project.id == project_id).options(joinedload(Project.owner))
        ).scalar_one_or_none()
        if not proj: abort(404)

        page = s.execute(
            select(Page).where(Page.id == page_id, Page.project_id == project_id)
        ).scalar_one_or_none()
        if not page: abort(404)

        topic = s.execute(
            select(Topic)
            .where(Topic.id == topic_id, Topic.page_id == page_id)
            .options(selectinload(Topic.codecells))
        ).scalar_one_or_none()
        if not topic: abort(404)

        cells = list(topic.codecells)

    return render_template(
        "code_workspace.html",
        project=proj, page=page, topic=topic, codecells=cells
    )

from flask import Response
@app.get("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/code/download")
@login_required
def topic_code_download(project_id, page_id, topic_id):
    require_project_member(project_id)
    lang = (request.args.get("lang") or "python").lower()
    with SessionLocal() as s:
        t = s.execute(
            select(Topic)
            .where(Topic.id == topic_id, Topic.page_id == page_id)
            .options(selectinload(Topic.codecells))
        ).scalar_one_or_none()
        if not t: abort(404)
        cells = list(t.codecells)

    src = "\n\n".join((c.code or "") for c in sorted(cells, key=lambda x: x.order_index))
    if lang == "cpp":
        header = f"// Project: {project_id} | Page: {page_id} | Topic: {topic_id}\n// Exported from BibApp\n\n"
        fn = f"topic_{topic_id}.cpp"
        mimetype = "text/x-c++src"
    else:
        header = f"# Project: {project_id} | Page: {page_id} | Topic: {topic_id}\n# Exported from BibApp\n\n"
        fn = f"topic_{topic_id}.py"
        mimetype = "text/x-python"
    out = header + src
    return Response(out, mimetype=mimetype, headers={"Content-Disposition": f"attachment; filename={fn}"})

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/paste-image")
@login_required
def topic_paste_image(project_id, page_id, topic_id):
    require_project_editor(project_id)
    file = request.files.get("file")
    if not file:
        return jsonify({"error":"no file"}), 400

    data = file.read()
    sha256 = hashlib.sha256(data).hexdigest()
    with SessionLocal() as s:
        blob = s.execute(select(Blob).where(Blob.sha256 == sha256)).scalar_one_or_none()
        if not blob:
            blob = Blob(sha256=sha256, filename=file.filename or "pasted.png", size=len(data), content_type=file.mimetype or "application/octet-stream", data=data)
            s.add(blob); s.flush()
        att = Attachment(topic_id=topic_id, blob_id=blob.id, label=file.filename or "pasted image")
        s.add(att); s.commit()
    url = url_for("serve_blob", sha_prefix=sha256[:2], sha=sha256, _external=False)
    return jsonify({"url": url})


# Open the Code Pad for a topic (read-only for viewers; editors can run/save)
@app.get("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/codepad")
@login_required
def topic_codepad(project_id: int, page_id: int, topic_id: int):
    # viewers can open; editors can also edit/execute
    require_project_member(project_id)

    with SessionLocal() as s:
        proj = s.execute(
            select(Project).where(Project.id == project_id)
        ).scalar_one_or_none()
        if not proj:
            abort(404)

        page = s.execute(
            select(Page).where(Page.id == page_id, Page.project_id == project_id)
        ).scalar_one_or_none()
        if not page:
            abort(404)

        topic = s.execute(
            select(Topic)
            .where(Topic.id == topic_id, Topic.page_id == page_id)
            .options(selectinload(Topic.codecells))
        ).scalar_one_or_none()
        if not topic:
            abort(404)

        # sort cells by order_index (if present); otherwise by id
        codecells = sorted(list(topic.codecells), key=lambda c: getattr(c, "order_index", c.id))

    # ENABLE_SERVER_EXEC should already be in your config/env
    return render_template(
        "codepad.html",
        project=proj,
        page=page,
        topic=topic,
        codecells=codecells,
        ENABLE_SERVER_EXEC=bool(os.getenv("ENABLE_SERVER_EXEC", "0") not in ("0", "", "false", "False"))
    )

@app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/codepad/cell")
@login_required
def codepad_cell_create(project_id: int, page_id: int, topic_id: int):
    require_project_editor(project_id)
    with SessionLocal() as s:
        # ensure topic exists
        exists = s.execute(
            select(Topic.id).where(Topic.id == topic_id, Topic.page_id == page_id)
        ).scalar_one_or_none()
        if not exists:
            abort(404)

        max_idx = s.execute(
            select(func.coalesce(func.max(TopicCodeCell.order_index), 0)).where(TopicCodeCell.topic_id == topic_id)
        ).scalar_one()

        new_cell = TopicCodeCell(topic_id=topic_id, code="", order_index=(max_idx or 0) + 1)
        s.add(new_cell)
        s.commit()
        cid = new_cell.id

    return redirect(url_for("topic_codepad", project_id=project_id, page_id=page_id, topic_id=topic_id) + f"#cell-{cid}")


@app.post("/project/<int:project_id>/delete")
@login_required
def project_delete(project_id: int):
    # Only the project owner can delete an entire project
    require_project_owner(project_id)

    with SessionLocal() as s:
        proj = s.execute(
            select(Project)
            .where(Project.id == project_id)
            .options(
                selectinload(Project.pages).selectinload(Page.topics).selectinload(Topic.codecells),
                selectinload(Project.pages).selectinload(Page.attachments),
                selectinload(Project.pages).selectinload(Page.annotations),
                selectinload(Project.memberships)
            )
        ).scalar_one_or_none()
        if not proj:
            abort(404)

        # Gather IDs for bulk deletes (defensive if cascades are not set)
        page_ids   = [p.id for p in proj.pages]
        topic_ids  = [t.id for p in proj.pages for t in p.topics]
        cell_ids   = [c.id for p in proj.pages for t in p.topics for c in t.codecells]

        # Delete in safe order (children → parents)
        if cell_ids:
            s.execute(delete(TopicCodeCell).where(TopicCodeCell.id.in_(cell_ids)))
        if topic_ids:
            s.execute(delete(Topic).where(Topic.id.in_(topic_ids)))
        if page_ids:
            s.execute(delete(Attachment).where(Attachment.page_id.in_(page_ids)))
            s.execute(delete(Annotation).where(Annotation.page_id.in_(page_ids)))
            s.execute(delete(Citation).where(Citation.page_id.in_(page_ids)))
            s.execute(delete(Page).where(Page.id.in_(page_ids)))

        # Project memberships (if you keep a separate table)
        s.execute(delete(ProjectMember).where(ProjectMember.project_id == project_id))

        # Finally, delete the project
        s.execute(delete(Project).where(Project.id == project_id))
        s.commit()

    # Notify collaborators that the project vanished
    try:
        socketio.emit("project_deleted", {"project_id": project_id}, to=f"project:{project_id}")
    except Exception:
        pass

    flash("Project deleted.", "success")
    return redirect(url_for("dashboard"))


# ---------- Citations ----------
def parse_authors(authors_str: str) -> List[str]:
    # "Last, First; Last, First M." -> list
    parts = [a.strip() for a in authors_str.split(";") if a.strip()]
    return parts or []

def fmt_apa(b: BibEntry) -> str:
    auth = parse_authors(b.authors)
    if auth:
        auth_txt = "; ".join(auth)
    else:
        auth_txt = ""
    year = f" ({b.year})." if b.year else "."
    rest = f" {b.title}."
    venue = f" {b.venue}." if b.venue else ""
    doi = f" https://doi.org/{b.doi}" if b.doi else ""
    url = f" {b.url}" if b.url and not b.doi else ""
    return f"{auth}{year}{rest}{venue}{doi}{url}".strip()

def fmt_mla(b: BibEntry) -> str:
    auth = parse_authors(b.authors)
    auth_txt = (auth[0] + ". ") if auth else ""
    title = f"“{b.title}.” " if b.title else ""
    venue = f"{b.venue}, " if b.venue else ""
    year = f"{b.year}. " if b.year else ""
    tail = f"doi:{b.doi}. " if b.doi else (f"{b.url} " if b.url else "")
    return (auth_txt + title + venue + year + tail).strip()

def fmt_chicago(b: BibEntry) -> str:
    auth = "; ".join(parse_authors(b.authors))
    year = b.year or ""
    t = f"“{b.title}.”" if b.title else ""
    venue = b.venue or ""
    doi = f" doi:{b.doi}" if b.doi else ""
    url = f" {b.url}" if b.url and not b.doi else ""
    return f"{auth}. {year}. {t} {venue}.{doi}{url}".strip()

def fmt_bibtex(b: BibEntry) -> str:
    # Minimal @misc entry
    fields = []
    if b.title: fields.append(f'title = {{{b.title}}}')
    if b.authors: fields.append(f'author = {{{b.authors}}}')
    if b.year: fields.append(f'year = {{{b.year}}}')
    if b.venue: fields.append(f'note = {{{b.venue}}}')
    if b.doi: fields.append(f'doi = {{{b.doi}}}')
    if b.url: fields.append(f'url = {{{b.url}}}')
    fields_str = ",\n  ".join(fields)
    key = b.key or f"entry{b.id}"
    return f"@misc{{{key},\n  {fields_str}\n}}"

@app.post("/project/<int:project_id>/citations/add")
@login_required
def citation_add(project_id):
    proj = require_project_editor(project_id)
    # require_project_editor(project_id)
    f = request.form
    page_id = request.form.get("page_id")
    page_id = int(page_id) if page_id else None
    data = dict(
        key=(f.get("key") or "").strip(),
        title=(f.get("title") or "").strip(),
        authors=(f.get("authors") or "").strip(),
        venue=(f.get("venue") or "").strip(),
        year=(f.get("year") or "").strip(),
        doi=(f.get("doi") or "").strip(),
        url=(f.get("url") or "").strip(),
        abstract=(f.get("abstract") or "").strip(),
        tags=(f.get("tags") or "").strip(),
    )
    with Session(engine) as s:
        c = BibEntry(project_id=project_id, page_id=page_id, **data)
        s.add(c); s.commit()
        cid = c.id
    socketio.emit("citation_added", {"project_id": project_id, "page_id": page_id, "citation_id": cid}, to=f"project:{project_id}")
    return redirect(url_for("project_view", project_id=project_id))

@app.get("/project/<int:project_id>/citations/export/<fmt>")
@login_required
def citations_export(project_id, fmt):
    proj = require_project_member(project_id)
    fmt = fmt.lower()
    with Session(engine) as s:
        entries = s.scalars(select(BibEntry).where(BibEntry.project_id == project_id)).all()
    if fmt == "apa":
        out = "\n".join(fmt_apa(b) for b in entries)
        mime = "text/plain"
    elif fmt == "mla":
        out = "\n".join(fmt_mla(b) for b in entries); mime = "text/plain"
    elif fmt == "chicago":
        out = "\n".join(fmt_chicago(b) for b in entries); mime = "text/plain"
    elif fmt == "bibtex":
        out = "\n\n".join(fmt_bibtex(b) for b in entries); mime = "text/plain"
    else:
        abort(400)
    return out, 200, {"Content-Type": mime}

# -------------------------
# Socket.IO events
# -------------------------
from flask_socketio import join_room
# routes_exec.py (or inside app.py)
import subprocess, threading, signal, shlex
from flask import Blueprint, request, jsonify, current_app
from flask_socketio import SocketIO, join_room, emit

# socketio: SocketIO = ...  # your existing SocketIO instance
# exec_bp = Blueprint("exec_bp", __name__)

# Track one running process per topic (simple approach)
RUNNING = {}  # key=(project_id,page_id,topic_id) -> Popen

# @socketio.on("join_topic")
# def on_join_topic(data):
#     room = f"topic:{data['project_id']}:{data['page_id']}:{data['topic_id']}"
#     join_room(room)

@socketio.on("code_stop")
def on_code_stop(data):
    key = (data["project_id"], data["page_id"], data["topic_id"])
    p = RUNNING.pop(key, None)
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=2)
        except: 
            p.send_signal(signal.SIGKILL)

# @app.post("/project/<int:project_id>/page/<int:page_id>/topic/<int:topic_id>/exec")
# def exec_server_code(project_id, page_id, topic_id):
#     if not current_app.config["EXEC_SERVER_ENABLED"]:
#         return jsonify({"ok": False, "error": "[server] Disabled. Ask project owner to enable server execution."}), 403

#     data = request.get_json(silent=True) or {}
#     code = (data.get("code") or "").strip()
#     lang = (data.get("lang") or "python").lower()
#     if not code:
#         return jsonify({"ok": False, "error": "No code"}), 400

    # choose interpreter
    if lang == "python":
        cmd = ["python", "-u", "-c", code]
    elif lang == "cpp":
        # naive demo: compile to /tmp/a.out and run it
        src = f"/tmp/run_{project_id}_{page_id}_{topic_id}.cpp"
        binp = f"/tmp/run_{project_id}_{page_id}_{topic_id}.out"
        open(src, "w").write(code)
        compile_cmd = ["g++", "-O2", "-std=c++17", src, "-o", binp]
        try:
            subprocess.check_output(compile_cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
        except subprocess.CalledProcessError as e:
            _emit_output(project_id, page_id, topic_id, "stderr", e.output)
            _emit_done(project_id, page_id, topic_id, returncode=1)
            return jsonify({"ok": True})
        cmd = [binp]
    else:
        return jsonify({"ok": False, "error": f"Unsupported lang: {lang}"}), 400

    room = f"topic:{project_id}:{page_id}:{topic_id}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    RUNNING[(project_id, page_id, topic_id)] = p

    def _pump(stream, label):
        for line in iter(stream.readline, ""):
            _emit_output(project_id, page_id, topic_id, label, line)
        stream.close()

    threading.Thread(target=_pump, args=(p.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=_pump, args=(p.stderr, "stderr"), daemon=True).start()

    def _waiter():
        rc = p.wait()
        RUNNING.pop((project_id, page_id, topic_id), None)
        _emit_done(project_id, page_id, topic_id, rc)

    threading.Thread(target=_waiter, daemon=True).start()
    return jsonify({"ok": True})

def _emit_output(project_id, page_id, topic_id, stream, data):
    socketio.emit("code_output",
        {"project_id": project_id, "page_id": page_id, "topic_id": topic_id, "stream": stream, "data": data},
        to=f"topic:{project_id}:{page_id}:{topic_id}",
    )

def _emit_done(project_id, page_id, topic_id, returncode):
    socketio.emit("code_done",
        {"project_id": project_id, "page_id": page_id, "topic_id": topic_id, "returncode": returncode},
        to=f"topic:{project_id}:{page_id}:{topic_id}",
    )

@socketio.on("exec:run")
def handle_exec(payload):
    ok, msg = _exec_is_allowed()
    if not ok:
        emit("exec:result", {"ok": False, "error": msg}, to=request.sid)
        return
    # ... run and emit result ...

@socketio.on("join_topic")
def on_join_topic(data):
    topic_id = int(data.get("topic_id", 0) or 0)
    if not topic_id: return
    join_room(f"topic:{topic_id}")
    socketio.emit("code_output", {"topic_id": topic_id,
                                  "stream":"status",
                                  "data":"🟢 joined stream\n"},
                  to=f"topic:{topic_id}")


@socketio.on("join_project")
def on_join_project(data):
    project_id = int(data["project_id"])
    if not current_user.is_authenticated: return
    try: require_project_member(project_id)
    except: return
    join_room(f"project:{project_id}")
    emit("presence", {"project_id": project_id, "user": current_user.email}, to=f"project:{project_id}")

@socketio.on("leave_project")
def on_leave_project(data):
    project_id = int(data["project_id"])
    leave_room(f"project:{project_id}")

@socketio.on("join_page")
def on_join_page(data):
    project_id = int(data["project_id"])
    page_id = int(data["page_id"])
    if not current_user.is_authenticated: return
    try: require_project_member(project_id)
    except: return
    join_room(f"page:{page_id}")
    emit("presence_page", {"page_id": page_id, "user": current_user.email}, to=f"page:{page_id}")

@socketio.on("page_update")
def on_page_update(data):
    # Optimistic concurrency: client sends html + expected revision
    project_id = int(data["project_id"])
    page_id = int(data["page_id"])
    html = data.get("content_html", "")
    client_rev = int(data.get("revision", 0))
    if not current_user.is_authenticated: return
    try: require_project_editor(project_id)
    except: return
    with Session(engine) as s:
        page = s.get(Page, page_id)
        if not page or page.project_id != project_id: return
        if client_rev != page.revision:
            emit("page_conflict", {"page_id": page_id, "server_revision": page.revision, "server_html": page.content_html})
            return
        page.content_html = html
        page.revision += 1
        s.commit()
        emit("page_broadcast", {"page_id": page_id, "html": html, "revision": page.revision, "user": current_user.email}, to=f"page:{page_id}")
@socketio.on("join_topic")
def on_join_topic(data):
    topic_id = int(data["topic_id"])
    join_room(f"topic:{topic_id}")

@socketio.on("join_topic")
def on_join_topic(data):
    project_id = int(data["project_id"])
    page_id = int(data["page_id"])
    topic_id = int(data["topic_id"])
    if not current_user.is_authenticated:
        return
    try:
        require_project_member(project_id)
    except:
        return
    join_room(f"topic:{topic_id}")
    emit("presence_topic", {"topic_id": topic_id, "user": current_user.email}, to=f"topic:{topic_id}")

@socketio.on("topic_update")
def on_topic_update(data):
    project_id = int(data["project_id"])
    page_id = int(data["page_id"])
    topic_id = int(data["topic_id"])
    md = data.get("content_md", "")
    client_rev = int(data.get("revision", 0))
    if not current_user.is_authenticated:
        return
    try:
        require_project_editor(project_id)
    except:
        return
    with SessionLocal() as s:
        t = s.get(Topic, topic_id)
        if not t or t.page_id != page_id:
            return
        if client_rev != t.revision:
            emit("topic_conflict", {"topic_id": topic_id, "server_revision": t.revision, "server_md": t.content_md})
            return
        t.content_md = md
        t.revision += 1
        s.commit()
        emit("topic_broadcast", {"topic_id": topic_id, "md": md, "revision": t.revision, "user": current_user.email}, to=f"topic:{topic_id}")


# -------------------------
# Minimal pages/templates
# -------------------------
# app.py
from flask_wtf.csrf import generate_csrf

@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="You don’t have permission to do that."), 403

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="The page you requested was not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Something went wrong on our side."), 500

@app.route("/about")
def about():
    return "Bibliography Tool — collaborative, condensed research pages."

# app.py
import os

port = int(os.environ.get("PORT", 8888))

if __name__ == "__main__":
    try:
        # If using Flask-SocketIO
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            allow_unsafe_werkzeug=True,  # dev server in prod (temporary)
        )
    except NameError:
        # Plain Flask
        app.run(host="0.0.0.0", port=port)
