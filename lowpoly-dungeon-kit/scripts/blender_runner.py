"""
blender_runner.py
=================
Locates a Blender runtime and executes pipeline scripts with it.

Two execution modes are supported and auto-detected:

``binary``
    A real ``blender`` executable (``$BLENDER_BIN`` or ``blender`` on PATH).
    Scripts are launched exactly like the marketplace-standard headless command::

        blender --background --python scripts/pipeline.py -- --asset wall

``module``
    The official ``bpy`` Python module (``pip install bpy``).  Blender runs
    in-process, which is how the shipped kit was produced inside a container
    that has no system Blender package.

The module mode needs the X11/OpenGL shared objects that Blender links against
even in background mode.  On a machine with those libraries present nothing is
needed; on a bare container :func:`ensure_stub_libraries` synthesises no-op
stubs (never called during background rendering) so ``import bpy`` succeeds.

CLI
---
    python scripts/blender_runner.py --check     # environment report
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
KIT_ROOT = SCRIPTS_DIR.parent
STUB_DIR = KIT_ROOT / "temp" / "xstubs"

# Shared objects Blender links against that are commonly absent from slim
# container images. Order matters only for readability.
STUB_LIBRARIES = [
    "libGL.so.1", "libICE.so.6", "libSM.so.6", "libXfixes.so.3",
    "libXi.so.6", "libXrender.so.1", "libxkbcommon.so.0",
]

# Symbols resolved through a catch-all preload library when the stubs above are
# not enough (Blender's GL/X11 usage in --background never invokes them).
STUB_CATCHALL = "libzstub.so"


# --------------------------------------------------------------------------- #
# Runtime detection
# --------------------------------------------------------------------------- #
def find_blender_binary() -> str | None:
    """Return a path to a ``blender`` executable, or ``None``."""
    env = os.environ.get("BLENDER_BIN")
    if env and Path(env).exists():
        return env
    for candidate in ("blender", "blender-4.2", "blender-4.1"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def bpy_available() -> bool:
    try:
        import bpy  # noqa: F401
        return True
    except Exception:
        return False


def _probe_bpy_in_subprocess(env_extra: dict | None = None) -> str | None:
    """Return the bpy version string if a child interpreter can import it."""
    code = "import bpy; print(bpy.app.version_string)"
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=180,
                              env={**os.environ, **(env_extra or {})})
    except Exception:
        return None
    if proc.returncode == 0:
        return proc.stdout.strip().splitlines()[-1]
    return None


def detect_runtime() -> dict:
    """Describe how the pipeline will talk to Blender (never raises for mode)."""
    binary = find_blender_binary()
    if binary:
        try:
            out = subprocess.run([binary, "--version"], capture_output=True,
                                 text=True, timeout=60).stdout.strip().splitlines()
        except Exception:
            out = []
        return {"mode": "binary", "path": binary,
                "version": out[0] if out else "unknown"}

    version = _probe_bpy_in_subprocess()
    if version is None:
        try:                      # bare container: synthesise GL/X11 stubs once
            stubs = ensure_stub_libraries()
            version = _probe_bpy_in_subprocess(stubs)
        except RuntimeError:
            version = None
    if version:
        return {"mode": "module", "path": sys.executable,
                "version": f"bpy {version}"}
    raise RuntimeError(
        "No Blender runtime found. Install one of:\n"
        "  * Blender 4.2+ binary and export BLENDER_BIN=/path/to/blender\n"
        "  * pip install bpy  (official Blender-as-a-Python-module build)")


def ensure_host_ready() -> None:
    """Guarantee that **this** interpreter can ``import bpy``.

    ``LD_PRELOAD`` must be set before a process starts, so when stub libraries
    are required we transparently re-exec the current script with the right
    environment. Scripts that import :mod:`recon` (which imports ``bpy`` at
    module level) must call this first.
    """
    if bpy_available():
        return
    if find_blender_binary():
        raise RuntimeError(
            "This script must run inside Blender; use blender_runner.run_script()")
    if os.environ.get("ASSET3D_REEXEC") == "1":
        raise RuntimeError("bpy still not importable after re-exec")
    stubs = ensure_stub_libraries(verbose=True)
    env = {**os.environ, **stubs, "ASSET3D_REEXEC": "1"}
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


# --------------------------------------------------------------------------- #
# Stub shared libraries (bare containers only)
# --------------------------------------------------------------------------- #
def _ldd_missing(binary_or_module: str) -> list[str]:
    """Return the list of 'not found' libraries for a shared object."""
    try:
        out = subprocess.run(["ldd", binary_or_module], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return []
    return [line.split("=>")[0].strip() for line in out.splitlines()
            if "not found" in line]


def ensure_stub_libraries(verbose: bool = False) -> dict | None:
    """Make ``import bpy`` work on hosts without X11/GL system libraries.

    Returns ``{"LD_LIBRARY_PATH": ..., "LD_PRELOAD": ...}`` when stubs were
    required, or ``None`` when the environment is already complete.
    """
    if bpy_available():
        return None

    # Fast path: a previously built stub set is reused as-is. Rebuilding on
    # every invocation is slow and races when several pipeline stages run
    # concurrently.
    cached = {"LD_LIBRARY_PATH": f"{STUB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}",
              "LD_PRELOAD": str(STUB_DIR / STUB_CATCHALL)}
    if (STUB_DIR / STUB_CATCHALL).exists():
        if _probe_bpy_in_subprocess(cached):
            os.environ.update(cached)
            return cached

    if shutil.which("gcc") is None:
        raise RuntimeError(
            "bpy is not importable and gcc is unavailable to build stub "
            "libraries; install libxrender1/libxi6/libgl1 or a Blender binary")

    STUB_DIR.mkdir(parents=True, exist_ok=True)
    def _compile(src: Path, out: Path, soname: str) -> None:
        """Compile atomically: never leave a half-written .so for other runs."""
        tmp = out.with_suffix(out.suffix + ".tmp")
        subprocess.run(["gcc", "-shared", "-fPIC", "-O0", f"-Wl,-soname,{soname}",
                        "-o", str(tmp), str(src)], check=True)
        tmp.replace(out)

    for lib in STUB_LIBRARIES:
        src = STUB_DIR / f"{lib}.c"
        src.write_text("void *PLACEHOLDER(void){return 0;}\n")
        _compile(src, STUB_DIR / lib, lib)

    catch_src = STUB_DIR / "libzstub.c"
    symbols: list[str] = []
    env = {"LD_LIBRARY_PATH": f"{STUB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}",
           "LD_PRELOAD": str(STUB_DIR / STUB_CATCHALL)}
    for attempt in range(80):
        catch_src.write_text(
            "".join(f"void *{s}(void){{return 0;}}\n" for s in symbols)
            + "void *PLACEHOLDER2(void){return 0;}\n")
        _compile(catch_src, STUB_DIR / STUB_CATCHALL, STUB_CATCHALL)
        probe = subprocess.run([sys.executable, "-c", "import bpy"],
                               capture_output=True, text=True,
                               env={**os.environ, **env})
        if probe.returncode == 0:
            if verbose:
                print(f"[blender_runner] stubs ready after {attempt + 1} passes")
            os.environ.update(env)   # children inherit it from here on
            return env
        msg = probe.stderr
        if "undefined symbol:" in msg:
            raw = msg.split("undefined symbol:")[1].split()[0].strip()
            sym = re.sub(r"[^A-Za-z0-9_]", "", raw)   # drop version suffixes / commas
            if not sym or sym in symbols:
                break
            symbols.append(sym)
        elif "error while loading shared libraries:" in msg:
            lib = msg.split("error while loading shared libraries:")[1].split(":")[0].strip()
            if not (STUB_DIR / lib).exists() and shutil.which("gcc"):
                (STUB_DIR / f"{lib}.c").write_text("void *PLACEHOLDER(void){return 0;}\n")
                _compile(STUB_DIR / f"{lib}.c", STUB_DIR / lib, lib)
            else:
                break
        else:
            break
    raise RuntimeError("failed to build stub shared libraries for bpy; "
                       "install the X11/GL packages instead")


def runtime_env(verbose: bool = False) -> dict:
    """Environment variables required to run scripts with Blender."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPTS_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    stubs = ensure_stub_libraries(verbose=verbose)
    if stubs:
        env.update(stubs)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


# --------------------------------------------------------------------------- #
# Script execution
# --------------------------------------------------------------------------- #
def run_script(script: str | Path, args: list[str] | None = None,
               *, timeout: int = 3600, verbose: bool = True,
               capture: bool = False) -> subprocess.CompletedProcess:
    """Execute a pipeline script under the detected Blender runtime.

    ``script`` is resolved relative to ``scripts/`` when a bare name is given.
    Arguments are appended after the ``--`` separator so Blender does not try
    to parse them itself.
    """
    script = Path(script)
    if not script.is_absolute():
        script = SCRIPTS_DIR / script
    runtime = detect_runtime()
    env = runtime_env(verbose=verbose)
    args = [str(a) for a in (args or [])]

    if runtime["mode"] == "binary":
        cmd = [runtime["path"], "--background", "--factory-startup",
               "--python-exit-code", "1", "--python", str(script), "--", *args]
    else:
        cmd = [sys.executable, str(script), *args]

    if verbose:
        print(f"[blender_runner] {runtime['mode']} :: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(KIT_ROOT), env=env, timeout=timeout,
                          capture_output=capture, text=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blender runtime probe / runner")
    ap.add_argument("--check", action="store_true", help="print an environment report")
    ap.add_argument("--run", metavar="SCRIPT", help="run a script under Blender")
    # Everything after the script name (or after `--`) is forwarded verbatim.
    ns, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    ns.args = extra

    if ns.check or not ns.run:
        try:
            info = detect_runtime()
        except RuntimeError as exc:
            print(f"[blender_runner] ERROR: {exc}")
            return 2
        print("Blender runtime report")
        print(f"  mode    : {info['mode']}")
        print(f"  path    : {info['path']}")
        print(f"  version : {info['version']}")
        try:
            stubs = ensure_stub_libraries(verbose=True)
            print(f"  stubs   : {'yes -> ' + stubs['LD_PRELOAD'] if stubs else 'not needed'}")
        except RuntimeError as exc:
            print(f"  stubs   : FAILED ({exc})")
        return 0

    proc = run_script(ns.run, ns.args)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
