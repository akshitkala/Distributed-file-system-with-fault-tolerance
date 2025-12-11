from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, json, math, hashlib, shutil, threading, time
from pathlib import Path

# ------------------ Constants ------------------
BASE_DIR = Path.cwd()
NODES_DIR = BASE_DIR / "nodes"
NAMENODE_DIR = BASE_DIR / "namenode"
FRONTEND_DIR = BASE_DIR / "frontend"
METADATA_FILE = NAMENODE_DIR / "metadata.json"
CHUNK_SIZE = 256 * 1024  # 256 KB
HEALER_INTERVAL = 3.0
REPLICATION = 2

# ------------------ Setup ------------------
app = FastAPI(title="Distributed File System Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ------------------ Helper Functions ------------------
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def ensure_dirs():
    NODES_DIR.mkdir(exist_ok=True)
    NAMENODE_DIR.mkdir(exist_ok=True)
    if not METADATA_FILE.exists():
        METADATA_FILE.write_text(json.dumps({"files": {}, "nodes": {}}, indent=2))


def load_meta():
    ensure_dirs()
    try:
        return json.loads(METADATA_FILE.read_text())
    except Exception:
        return {"files": {}, "nodes": {}}


def save_meta(meta):
    METADATA_FILE.write_text(json.dumps(meta, indent=2))


# ------------------ DFS Core ------------------
class NameNode:
    def __init__(self):
        self.meta = load_meta()
        self.lock = threading.Lock()

    def healthy_nodes(self):
        return [nid for nid, n in self.meta["nodes"].items() if n["alive"]]

    def choose_nodes(self, k, exclude=None):
        exclude = set(exclude or [])
        candidates = [n for n in self.healthy_nodes() if n not in exclude]
        return candidates[:k]

    def register_file(self, file_name, num_chunks):
        with self.lock:
            file_id = f"{file_name}-{int(time.time()*1000)}"
            self.meta["files"][file_id] = {
                "file_name": file_name,
                "num_chunks": num_chunks,
                "chunks": {}
            }
            save_meta(self.meta)
            return file_id

    def record_chunk(self, file_id, idx, checksum, replicas):
        with self.lock:
            self.meta["files"][file_id]["chunks"][str(idx)] = {
                "checksum": checksum,
                "replicas": list(replicas)
            }
            save_meta(self.meta)


nn = NameNode()


def add_node(node_id):
    node_path = NODES_DIR / node_id
    node_path.mkdir(parents=True, exist_ok=True)
    nn.meta["nodes"][node_id] = {"path": str(node_path), "alive": True}
    save_meta(nn.meta)


def remove_node(node_id):
    if node_id not in nn.meta["nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")
    node_path = Path(nn.meta["nodes"][node_id]["path"])
    shutil.rmtree(node_path, ignore_errors=True)
    del nn.meta["nodes"][node_id]
    save_meta(nn.meta)


def set_node_alive(node_id, alive: bool):
    if node_id not in nn.meta["nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")
    nn.meta["nodes"][node_id]["alive"] = alive
    save_meta(nn.meta)


# ------------------ Endpoints ------------------

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = FRONTEND_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/status")
def status():
    return {
        "nodes": nn.meta.get("nodes", {}),
        "files": list(nn.meta.get("files", {}).keys()),
    }


@app.post("/add_node")
def api_add_node():
    i = 1
    while True:
        nid = f"n{i}"
        if nid not in nn.meta["nodes"]:
            break
        i += 1
    add_node(nid)
    return {"status": "added", "node_id": nid}


@app.delete("/remove_node/{node_id}")
def api_remove_node(node_id: str):
    remove_node(node_id)
    return {"status": "deleted", "node_id": node_id}


@app.post("/toggle_node/{node_id}")
def api_toggle_node(node_id: str):
    if node_id not in nn.meta["nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")
    alive = not nn.meta["nodes"][node_id]["alive"]
    set_node_alive(node_id, alive)
    return {"node_id": node_id, "alive": alive}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    size = len(data)
    num_chunks = math.ceil(size / CHUNK_SIZE)
    file_id = nn.register_file(file.filename, num_chunks)
    healthy = nn.healthy_nodes()
    if len(healthy) < REPLICATION:
        raise HTTPException(status_code=400, detail="Not enough healthy nodes")

    for i in range(num_chunks):
        chunk = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        checksum = sha256_bytes(chunk)
        chosen = nn.choose_nodes(REPLICATION)
        for nid in chosen:
            node_path = Path(nn.meta["nodes"][nid]["path"])
            (node_path / file_id).mkdir(exist_ok=True)
            (node_path / file_id / f"chunk-{i:06d}").write_bytes(chunk)
        nn.record_chunk(file_id, i, checksum, chosen)
    return {"status": "uploaded", "file_id": file_id, "chunks": num_chunks}


@app.get("/download")
def download_file(file_id: str):
    if file_id not in nn.meta["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    finfo = nn.meta["files"][file_id]
    out = bytearray()
    healthy = nn.healthy_nodes()
    for i in range(finfo["num_chunks"]):
        cinfo = finfo["chunks"].get(str(i))
        if not cinfo:
            continue
        for nid in cinfo["replicas"]:
            if nid in healthy:
                node_path = Path(nn.meta["nodes"][nid]["path"])
                data = (node_path / file_id / f"chunk-{i:06d}").read_bytes()
                if sha256_bytes(data) == cinfo["checksum"]:
                    out.extend(data)
                    break
    out_path = BASE_DIR / f"downloaded_{finfo['file_name']}"
    out_path.write_bytes(out)
    return FileResponse(str(out_path), filename=finfo["file_name"])


@app.get("/verify")
def verify_file(file_id: str):
    if file_id not in nn.meta["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    finfo = nn.meta["files"][file_id]
    healthy = set(nn.healthy_nodes())
    ok = True
    for i, cinfo in finfo["chunks"].items():
        good = any(
            (Path(nn.meta["nodes"][nid]["path"]) / file_id / f"chunk-{int(i):06d}").exists()
            for nid in cinfo["replicas"]
            if nid in healthy
        )
        if not good:
            ok = False
    return {"file_id": file_id, "healthy": ok}

@app.delete("/delete_file/{file_id}")
def delete_file(file_id: str):
    if file_id not in nn.meta["files"]:
        raise HTTPException(status_code=404, detail="File not found")

    finfo = nn.meta["files"][file_id]
    for idx, cinfo in finfo["chunks"].items():
        for nid in cinfo["replicas"]:
            if nid in nn.meta["nodes"]:
                node_path = Path(nn.meta["nodes"][nid]["path"]) / file_id
                if node_path.exists():
                    shutil.rmtree(node_path, ignore_errors=True)

    del nn.meta["files"][file_id]
    save_meta(nn.meta)
    return {"status": "deleted", "file_id": file_id}

@app.get("/file_info/{file_id}")
def file_info(file_id: str):
    if file_id not in nn.meta["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    finfo = nn.meta["files"][file_id]
    return {
        "file_id": file_id,
        "file_name": finfo["file_name"],
        "num_chunks": finfo["num_chunks"],
        "chunks": finfo["chunks"],
        "replication_factor": REPLICATION,
    }


# ------------------ Background Self-Healer ------------------
def healer():
    while True:
        rf = REPLICATION
        healthy = nn.healthy_nodes()
        for file_id, finfo in nn.meta.get("files", {}).items():
            for idx, cinfo in finfo["chunks"].items():
                reps = [r for r in cinfo["replicas"] if r in healthy]
                if len(reps) < rf and reps:
                    src = reps[0]
                    src_path = Path(nn.meta["nodes"][src]["path"]) / file_id / f"chunk-{int(idx):06d}"
                    if not src_path.exists():
                        continue
                    data = src_path.read_bytes()
                    new_nodes = [n for n in healthy if n not in reps][: rf - len(reps)]
                    for nid in new_nodes:
                        dest = Path(nn.meta["nodes"][nid]["path"]) / file_id
                        dest.mkdir(exist_ok=True)
                        (dest / f"chunk-{int(idx):06d}").write_bytes(data)
                        reps.append(nid)
                    nn.record_chunk(file_id, int(idx), cinfo["checksum"], reps)
        time.sleep(HEALER_INTERVAL)


threading.Thread(target=healer, daemon=True).start()
