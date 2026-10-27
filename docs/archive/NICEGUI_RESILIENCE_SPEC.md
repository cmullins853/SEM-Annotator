# NiceGUI Resilience Spec: refresh survival and session resume

**Source:** the Synplex Data Analysis GUI (`brendenw1/synplexdataanalysis`),
as of 2026-10-27, NiceGUI 3.16+.
**Target:** a new single-user, local NiceGUI app that runs ML models over SEM
images to measure fiber diameters. NiceGUI handles display and user input; the
analysis backend is a separate engine the GUI calls into.

This spec covers only the web-facing machinery that keeps work alive through
browser refreshes, backgrounded tabs, closed tabs, slow connections and server
restarts. It doesn't describe Synplex's analysis features.

### How to read it

- **REQ-xx** items are requirements. Each one either is implemented in Synplex,
  with a `Ref:` naming the file and function, or is marked
  **[NEW]**. **[NEW]** marks guidance for the ML app that Synplex never needed
  or never built, mostly resuming a batch interrupted mid-run. Synplex hasn't
  proven these, so test them first.
- §8 lists the failures Synplex actually hit. Each line in it explains why a
  requirement exists.
- Paths in `Ref:` are relative to `src/synplex/`. The repo layout keeps the
  page in `gui/app.py`, widgets in `gui/components.py`, server state in
  `gui/state.py` and the restart store in `session_store.py`.

---

## 1. What can go wrong (the failure model)

Design against every one of these. Synplex hit all except F7.

| # | Event | What NiceGUI does | What breaks if the app is naive |
|---|---|---|---|
| F1 | User presses refresh | Old client is deleted once `reconnect_timeout` runs out; `index()` runs again for a new client | Every local variable of the old page is gone; a running job's "on done" code writes into deleted widgets and **is silently dropped** |
| F2 | Tab backgrounded / laptop sleeps | Browser throttles the socket; after `reconnect_timeout` (**default 3 s**) the client is reaped | Same as F1, but the user didn't do anything |
| F3 | Second tab opened | A second, independent client | Two Run buttons, so two concurrent jobs over the same output tree and global state |
| F4 | Tab closed while a dialog is awaited | Dialog is deleted; since 3.16 `await dialog` returns `None` (older versions hung forever) | Code after the `await` treats "gone" as an answer, or touches a dead page. Older versions hung while holding resources (e.g. an open DB) |
| F5 | Event flood (slider drag, rapid edits) or huge payloads | Websocket backs up and times out; browser reconnects as a **new page** | Looks like a random refresh (F1) with no error in the terminal |
| F6 | Server restart / crash | All memory gone | Everything not on disk is lost, so the analysis has to be re-run |
| F7 | Server or worker dies **mid-job** | Same as F6, during work | Partial outputs on disk with no record of what is finished. **[NEW]** for the ML app; Synplex jobs are short (~40 s) |

The fix is three tiers of state plus disposable pages:

| Tier | Lives in | Survives | Holds |
|---|---|---|---|
| Page view | `index()` locals | nothing | widget handles, what this page last rendered (cursor / generation) |
| Run state | server-scope objects | F1–F5 | job status, progress, log, the loaded result, user's edits |
| Session | one JSON file on disk | F6 | pointers to outputs + user's edits (no bulk data) |
| Outputs | the output folder (+ DB) | everything | the actual results, written by the backend |

---

## 2. Server-owned state

**REQ-01 — Run state lives at server scope, never in the page function.**
Create the state object(s) once, in `main()` before `@ui.page` (or at module
level), so every `index()` call shares them. A refresh re-runs `index()` but
reuses the state.
Ref: `gui/app.py::main` creates `config = PipelineConfig()`,
`result = PipelineResult()`, plus `schedule`, `tracking_meta`,
`channel_windows`, `replot_state`, `rampfit_state` and `tc_state` before
`index()` is defined.

**REQ-02 — Run state holds no UI elements.** No `ui.*` object, no bound method
of one, no closure over one. State is plain data: str, numbers, lists,
dicts, paths, DataFrames/arrays.
Ref: `gui/state.py::PipelineResult` docstring ("Nothing here may hold a
reference to a UI element").

**REQ-03 — Handlers mutate state; they don't paint results.** A click
handler changes the state and returns. Rendering happens in the page's
reconcile loop (§3). One exception: a handler may give immediate feedback
(`ui.notify`) to the page that clicked, as long as it checks that page still
exists (REQ-30).
Ref: `gui/app.py::handle_run` — "No UI is touched from here on."

**REQ-04 — The state carries a `generation` counter and a status enum.**
`status ∈ {idle, running, complete, error}` ([NEW] add `cancelled` and
`interrupted` for the ML app). `generation` is incremented each time a new
result lands. Pages compare it with the generation they last rendered to
decide whether to rebuild the results area.
Ref: `gui/state.py::PipelineResult.generation`, `.status`.

**REQ-05 — Per-page view state is the only thing in `index()`.** A small
dict like `{"generation": -1, "status": None, "log_total": 0}` records
what this page has shown. Widget handles also live here. A refresh throws
this away, and that's harmless.
Ref: `gui/app.py::index`, `page_view`.

**REQ-06 — Registries of widget handles are per page.** Any dict that maps
keys to elements (tab refs, per-item refresh callbacks, input refs) must be
created inside `index()`, not shared.
Ref: **Synplex gets this partly wrong.** `file_refs` and `channel_refresh`
are server scope, so with two tabs open they point at whichever tab built
last (finding B5, still open). `tab_refs` was moved into `index()`. Don't
copy the shared ones.

---

## 3. The reconcile loop (how a page shows state)

**REQ-10 — Every page runs a reconcile function on a timer.**
`ui.timer(0.5, reconcile)` is created inside `index()`. The timer belongs to
the client and stops when that client is deleted. Call `reconcile()` once
before creating the timer so the first paint is immediate.
Ref: `gui/app.py::index`, `_reconcile()` then `ui.timer(0.5, _reconcile)`.

**REQ-11 — One code path covers first load, refresh, and following a job
another page started.** Don't write separate "restore on load" code. A
restore block that runs once at build time runs *before* the job finishes and
never fires again. That was the original refresh bug (§8, H1).
Ref: comment above `page_view` in `gui/app.py`.

**REQ-12 — The reconcile function never raises.** Wrap the body in
`try/except Exception`, log, and continue. If the timer callback dies, the
page stops syncing for good.
Ref: `gui/app.py::_reconcile` / `_reconcile_inner`.

**REQ-13 — Reconcile is cheap and change-driven.**
- Idempotent setters (`progress.set_value`, `button.enable()`) are fine on
  every tick.
- Anything that sends class/prop changes or rebuilds elements runs only when
  the value differs from `page_view`.
- Rebuild the results area only when `generation` changed.
- Resolve expensive lookups (file existence etc.) only when their input
  changed.

Ref: `_set_status_chip` (only on status change), `_set_step` (only on
state/text change), `_update_stepper` (re-resolves the file only when
`config.raw_file` changes).

**REQ-14 — Reconcile drives every run-dependent control.** That includes the
Run button's enabled and loading state, the status text, the progress bar, the
stepper or checklist, the log, and the results area. A refreshed page mid-job
must show the button disabled and progress live.
Ref: `_reconcile_inner` run-button block.

**REQ-15 — Notifications about a result fire only for a transition this page
watched.** On the first pass (`page_view["status"] is None`), a result that
already exists is labelled "restored" rather than announced as "finished".
Error toasts fire only on a transition into `error`.
Ref: `_reconcile_inner`, `first_pass` and the `restored` flag.

**REQ-16 — Preserve the user's place across rebuilds.** When the results
area rebuilds, read the active tab first and restore it. Open a tab by
default, because Quasar otherwise shows a blank panel.
Ref: `gui/app.py::_refresh_gallery(switch_to=...)` reads
`tab_refs["tabs"].value`; `components.plot_gallery(active_tab=...)`.

---

## 4. Running work in the background

**REQ-20 — The heavy work runs off the event loop.** Use
`await asyncio.to_thread(fn)` (or `nicegui.run.io_bound`) for work that
releases the GIL. PyTorch/ONNX inference and file I/O do. Use
`run.cpu_bound` or a separate process for pure-Python CPU work. Loading
results afterwards (reading CSVs, images) also goes through `to_thread`.
Ref: `handle_run` → `await asyncio.to_thread(_run_sync)`, then
`await asyncio.to_thread(_load_cleaned, ...)`.

**REQ-21 — The job's completion doesn't depend on the page that started
it.** NiceGUI doesn't cancel a handler's task when its client is deleted
(verified in 3.18's `Client.delete`). The `await` in `handle_run` therefore
still resumes after a refresh, and because it only writes to server state
(REQ-03), the result lands. **[NEW]** For long ML batches, prefer to start the
job as a server-level task (`nicegui.background_tasks.create(...)`) or inside
the backend itself. That way nothing ties it to a page-level click handler.

**REQ-22 — One job at a time, enforced in state, not in the button.**
`handle_run` refuses if `state.status == "running"`. Disabling the clicked
button only protects that one page, and a refreshed or second tab has its own
enabled button (F3).
Ref: top of `handle_run`.

**REQ-23 — Publish results atomically: data first, flags last.** A reconcile
tick can run at any `await`. Fill in every result field first, then set
`progress = 1.0`, then `status = "complete"`, then `generation += 1`. A page
must never see "complete" with a half-filled result.
Ref: the "Publish the finished result as one step" block in `handle_run`.

**REQ-24 — Reset per-experiment state only after the new job succeeds.**
User edits that belong to the previous result (windows, annotations,
calibration, trial name) are cleared when a *different* input succeeds.
Re-running the same input keeps them. A failed run leaves the previous result
on screen with its matching edits intact. Actions such as "record" or "export"
check `status == "complete"`, so they never mix a failed new input with old
data.
Ref: `_reset_experiment_state`, called in `handle_run` after success when
`result.source_file` changed; `handle_record` checks `status != "complete"`
(R-G1/R-G2).

**REQ-25 — Pin what a long handler works on.** If a handler awaits (a dialog,
a computation), copy the state it needs into locals first. Another job may
replace the state during the await.
Ref: `handle_record` — `cleaned_df, source_file = result.cleaned_df, result.source_file`.

### Log and progress channel

**REQ-26 — The worker writes progress and log lines into state, never into
widgets.** Building or mutating NiceGUI elements off the event loop is unsafe,
and those writes are lost once the page is gone.
Ref: `gui/app.py::_PipelineLogStream.write` appends to `result.log_lines`.
Progress is parsed from `[n/4]` markers into `result.progress`.

**REQ-27 — The log buffer is bounded and cursor-based.** Keep the last N lines
(Synplex: 500) plus a monotonically increasing `log_total`. Each page tracks
its own `log_total` and appends only the difference. If the difference is
negative, a new run reset the buffer, so clear the widget and replay. Trimming
the front of the buffer never makes a page skip or duplicate lines.
Ref: `PipelineResult.log_lines` / `log_total`; log block in `_reconcile_inner`.

**REQ-28 — Capture only the worker's output.** `contextlib.redirect_stdout` is
process-wide. If you capture `print()`, record the worker thread's id and pass
other threads' writes through to the real stdout.
Ref: `_PipelineLogStream.claim_current_thread`.
**[NEW]** Better for the ML app: give the backend an explicit
`on_progress(done, total, msg)` callback or a `logging` handler instead of
scraping stdout.

**REQ-29 — Force the non-interactive matplotlib backend before importing
analysis code** if the backend draws figures server-side:
`matplotlib.use("Agg")` at the top of the GUI module.
Ref: top of `gui/app.py`.

---

## 5. Connections, dialogs, and pages that disappear

**REQ-30 — Every `await` in a handler is followed by a "page gone?" check.**
Capture `client = ui.context.client` at the start of the handler. After any
await (above all a dialog), `if client.id not in Client.instances: return`.
Don't notify or touch UI on a dead page.
Ref: `gui/app.py::_page_gone`; used in `handle_upload`, `handle_record`,
`_offer_resume`.

**REQ-31 — A closed page isn't an answer.** With NiceGUI ≥ 3.16, `await dialog`
returns `None` both for Esc/backdrop and for a deleted page. Treat `None` as
"not now" and leave everything as it was. Don't treat it as "No"/"Cancel"
when that answer destroys something.
Ref: `handle_upload` (closed mid-overwrite-confirm keeps the old file,
R-G5); `_offer_resume` (Esc keeps the session, R-G3). Use `dlg.submit(value)`
on the buttons so the explicit answers are distinguishable from `None`.

**REQ-32 — Don't hold resources across a dialog.** Open the DB, file handle or
lock just for the read before the dialog. Close it, await the user, then open
it again for the write.
Ref: `handle_record` (R-G5).

**REQ-33 — Pin `nicegui>=3.16`.** Older versions hang forever on `await dialog`
after the page is deleted.
Ref: `pyproject.toml`.

**REQ-34 — Raise `reconnect_timeout`.** `ui.run(..., reconnect_timeout=30.0,
reload=False)`. The 3 s default reaps a backgrounded tab that the browser
throttles (F2). With a longer grace period, the same client reconnects with
its page intact.
Ref: `ui.run` at the end of `gui/app.py::main`.

---

## 6. Payload and event discipline (avoiding F5)

Synplex's "random refreshes" were really websocket timeouts caused by too much
data or too many events. The rules generalise directly to images.

**REQ-40 — Full resolution for every number, reduced resolution for every
pixel.** On-screen plots get downsampled data. Calculations and saved figures
always use the full data.
Ref: `components._downsample_lttb` (Largest-Triangle-Three-Buckets, ≤ 2500
points per trace, keeps spikes that a stride would lose); `_display_series`.
Result: 17 MB → 2 MB per gallery refresh, independent of data length.
**[NEW]** For SEM images: never push full-resolution images through the
websocket (`ui.image` with base64/PIL or `ui.interactive_image` with large
sources). Serve files from disk via `app.add_media_files` /
`app.add_static_files` and show downscaled thumbnails or previews. Draw
detected fibers/diameters as a light overlay (SVG content on
`ui.interactive_image`, or a downsampled Plotly layer). Compute histograms from
the measurement table, not from pixels.

**REQ-41 — Build tab panels lazily.** Quasar keeps every panel in the DOM, so
eager building sends every tab's content even if it's never opened. Build each
panel body the first time its tab activates.
Ref: `components.plot_gallery`, `builders` / `_ensure_built` /
`tabs.on_value_change`.

**REQ-42 — Update figures in place; don't rebuild containers.** For an edit
that affects one item, refresh that item's figure
(`plot.update_figure(fig)`) and fields through a per-item refresh callback.
Don't clear and rebuild the whole results area.
Ref: `_render_tracking_panel`, `registry[(sensor, channel)]()`, `_rebuild()`.
(Known gap R-G8: one drag still rebuilds a figure 3× and some actions rebuild
the whole gallery. Do better.)

**REQ-43 — Keep the user's zoom across redraws.** Set a stable Plotly
`uirevision` per figure. Bump `selectionrevision` on every build so an old
drag-selection box doesn't reappear.
Ref: `_tracking_figure` (`uirevision=f"tracking-{sensor}-{channel}"`),
`_apply_drag_mode`.

**REQ-44 — No continuous-event controls bound to expensive handlers.** A
slider whose `on_change` did disk I/O and a figure rebuild per drag tick
flooded the loop, timed out the socket and forced a reload. Use steppers or
number fields (one event per change), or throttle/debounce, or update on
release only.
Ref: Tracking "Time offset (min)" `ui.number` stepper (CHANGELOG, "Tracking tab
no longer loses work on a page reload").

**REQ-45 — Cache derived data that redraws need.** Keep cached data in state
under keys that are clearly derived (Synplex: a leading `_`, e.g.
`_sp_cache`), invalidate it when its source changes, and **never persist it**
(REQ-53).
Ref: `components._tracking_figure` / `_sp_cache`;
`session_store._strip_caches`.

**REQ-46 — Cache-bust regenerated files.** When a file on disk is rewritten
under the same name, add its mtime to the URL (`?t=<mtime_ns>`) so the browser
doesn't show the stale cached copy.
Ref: `components._zoomable_image` (U-11).

**REQ-47 — Don't snapshot the filesystem at page build.** Lists of input files
(or models, or batches) are re-read when the control opens or on an explicit
Refresh. One resolver turns any user-typed path into a real file, and the UI
shows the resolved path live, before Run.
Ref: `components.file_config_tile._refresh_options`;
`utils/data_io.find_input_file` / `input_path_candidates`.

**REQ-48 — Number fields must not round stored values.** In NiceGUI,
`ui.number(format="%.1f")` rewrites the *value* to the formatted precision on
blur (`101.325` → `101.3`). Use `format=` only on integer fields or where
truncation is intended.
Ref: finding A3. Physical-value fields in `components.py` have no `format`.

---

## 7. Session persistence and resume (surviving F6)

### What is saved

**REQ-50 — One session file, one active working set.** A single JSON file
under the app's base directory. It is not a history: finished/recorded work
belongs in the outputs and the DB.
Ref: `session_store.py`, `<base>/.synplex_session.json`.

**REQ-51 — The session stores pointers and user edits, not bulk data.** It
holds the output directory paths, the source input path, and everything the
user built up by hand (windows/annotations, per-item settings, trial identity,
labels). Bulky results are re-loaded from the output folder on resume.
Ref: `session_store.build_payload` (`dirs`, `source_file`, `schedule`,
`tracking_meta`, `channel_windows`); DataFrame deliberately excluded.
**[NEW]** For the ML app the session typically holds: batch/output dir,
input image list (or the folder plus a manifest pointer), model id and weights
hash, inference parameters, and user edits. User edits include scale-bar
calibration, ROI/exclusion masks as paths or polygons, rejected/accepted
detections, and manual diameter corrections.

**REQ-52 — The payload is versioned.** It has a `version` field. On load, an
unknown version is ignored (no resume offered), not guessed at. Bump the
version whenever the shape changes incompatibly.
Ref: `session_store.SESSION_VERSION` (2).

**REQ-53 — The payload is lossless JSON and contains no caches.** Strip
derived/cached keys before writing. Convert non-string dict keys (Synplex: int
channel numbers) to strings on save and back on load, and drop unreadable
keys instead of failing.
Ref: `_strip_caches`, `_encode_channel_map` / `_decode_channel_map`;
`tests/test_session_store.py::test_round_trip_restores_int_channels`.

### How it is written

**REQ-54 — Write atomically.** Write to a temp file in the same directory, then
`os.replace` it over the session file. An interrupted write never leaves a
truncated session.
Ref: `session_store.save_session` (`tempfile.mkstemp` + `os.replace`).

**REQ-55 — Saving never raises.** Failing to save the session must not take
down the user's current work. Catch, log, return `None`.
Ref: `save_session`.

**REQ-56 — Save explicitly at milestones, and autosave on change from
reconcile.** Save explicitly after a job completes and after each major
action. Also have every reconcile tick (while a result is loaded) build the
payload, serialise it with `sort_keys=True`, compare it with the last saved
snapshot, and write only if it differs. Edits are then saved within about a
second however they were made (typed, dragged, auto-filled, pasted), and no
handler has to remember to save.
Ref: `gui/app.py::_save_session(only_if_changed=True)`, the `_last_saved`
snapshot, the end of `_reconcile_inner`; `session_store.build_payload` is
public for this reason (U-10).

### How it is offered back

**REQ-57 — Validate before offering.** `load_session` returns `None` if the
file is missing, unreadable or the wrong version, if the output directory it
points to is gone, or if the key output file inside it is missing. Don't offer
a session that will only fail.
Ref: `session_store.load_session`, `find_cleaned_csv`.
(Known gap R-G9: valid JSON that isn't a dict breaks loading on every start.
Check `isinstance(payload, dict)`.)

**REQ-58 — Resume is offered, never automatic.** On startup, if a valid
session exists and nothing is loaded yet, show a dialog: name, date, output
path, **Resume** / **Start fresh**. Silently reloading an old experiment is how
stale edits get recorded against new data.
Ref: `gui/app.py::_offer_resume`, triggered by `ui.timer(0.3, _offer_resume,
once=True)` inside `index()`.

**REQ-59 — Offer once per server start, not once per page.** A server-scope
flag (`_resume_state["offered"]`) stops every refresh or new tab from asking
again. Skip the offer if a result is already loaded.

**REQ-60 — Only an explicit answer consumes the offer.**
- **Start fresh**: delete the session file (`clear_session`).
- **Resume**: load the session.
- **Esc / backdrop / page closed / task cancelled**: reset the flag to `False`
  so the next page asks again. Keep the file.

Ref: `_offer_resume` (`except asyncio.CancelledError: offered = False; raise`;
`if _page_gone(client) or answer is None: offered = False`; R-G3, U-8).

**REQ-61 — Resume restores through the normal channel.**
1. Load bulk data off the loop (`asyncio.to_thread`).
2. Restore user edits into the server-scope dicts (`.clear()` + `.update()` so
   existing references stay valid).
3. Point the input selector at the saved source.
4. Set result fields, then `status = "complete"` and `generation += 1`.

Every open page then rebuilds from its own reconcile tick. There's no special
"resumed" rendering path.
Ref: tail of `_offer_resume`.

**REQ-62 — Resumed and re-run sessions start in a safe mode.** Anything with
side effects that are hard to undo (Synplex: writing to the run database)
starts *off* after Run, after Resume and after each write. The user turns it
on again deliberately.
Ref: `tracking_meta["record"] = False` in `handle_run`, `_offer_resume`,
`handle_record`.

**REQ-63 — Clear user edits that don't belong to the restored input.** This is
the same reset as REQ-24. Choose the trial identity again from the new input
and don't carry it over.
Ref: `_reset_experiment_state`.

---

## 8. Pitfalls and history (what broke and why)

Each line is a real Synplex incident. The "Rule" column points back to the
requirement it produced.

| # | Symptom the user saw | Actual cause | Fix | Rule |
|---|---|---|---|---|
| H1 | Refresh mid-run: page came back Idle with no plots, and the run had to be repeated, though it had actually finished | Completion code wrote into the *old* page's widgets (deleted after 3 s; writes silently dropped). The new page's restore check ran once at build time, before the run ended | Server-owned `PipelineResult` + per-page reconcile timer + `generation` | REQ-01–04, 10–11 |
| H2 | Backgrounded tab "lost everything" | Browser throttled the socket; the 3 s `reconnect_timeout` reaped the client | `reconnect_timeout=30` | REQ-34 |
| H3 | Dragging the Tracking offset slider made the page reload and wiped entered windows | Slider fired an event per tick, each doing disk I/O + figure rebuild, so the socket timed out and the browser reconnected as a new page. Windows lived in the page closure | Stepper field, cached curve, edits hoisted to server scope, per-channel refresh | REQ-01, 42, 44, 45 |
| H4 | Tab "hung" when returning to it on long datasets | Every gallery refresh pushed 17 MB (68 MB for 24 h) of full-resolution traces, including tabs never opened | LTTB downsampling + lazy tab panels: a flat ~2 MB | REQ-40, 41 |
| H5 | A refreshed page could start a second run during the first | Only the clicking page's button was disabled | Server-wide `status == "running"` guard; reconcile disables Run everywhere | REQ-14, 22 |
| H6 | Rare UI corruption / stray output in the log | `ui.log.push` called from the worker thread; process-wide `redirect_stdout` captured the event loop's prints too | Worker appends to state; thread-owned stream | REQ-26, 28 |
| H7 | Server restart: all work gone, full re-run needed | Nothing on disk | Session file + Resume prompt | REQ-50–61 |
| H8 | Edits made after the last run were lost on restart | Session saved only at run end / Compute / Record | Autosave on change from reconcile (snapshot compare) | REQ-56 |
| H9 | Closing a tab with "Replace record?" open logged errors; closing it with the Resume prompt open meant Resume was never offered again until restart | Code after `await dialog` acted on a dead page; the "offered" flag was set even though nobody answered | `_page_gone` checks; reset flag on no answer / cancel | REQ-30, 31, 60 |
| H10 | Pressing Esc on the Resume prompt deleted the saved session | `None` (dismiss) treated as "Start fresh" | Only an explicit `submit("fresh")` clears | REQ-31, 60 |
| H11 | On NiceGUI < 3.16, Record hung forever with the DB open after a tab close | Awaited dialog never resolved on a deleted page; DB was open across the await | Pin ≥ 3.16; open DB only around read and write | REQ-32, 33 |
| H12 | Analysing file B after A recorded B's numbers under A's name/windows | Per-experiment state never reset between runs | Reset on a new source file | REQ-24, 63 |
| H13 | A *failed* run of B left A's plots on screen but B's name/state, and Record accepted it | Reset ran before the job instead of after success | Reset only on success; actions require `complete` | REQ-24 |
| H14 | 101.325 stored as 101.3 | `ui.number(format="%.1f")` rewrites the value | No `format=` on physical values | REQ-48 |
| H15 | Files copied into Data/ didn't appear; typed paths "not found" | Option list read once at page build; ad-hoc path resolution | Refresh on open; one resolver; live resolved-path line | REQ-47 |
| H16 | Re-rendered preview looked unchanged | Same URL, so browser cache | mtime query string | REQ-46 |
| H17 | Results area blank after a run until a tab was clicked; rebuilds threw the user back to tab 1 | Quasar selected no tab; rebuild didn't remember the active one | Default + remembered active tab | REQ-16 |
| H18 | Empty / corrupt input showed `Pipeline error: None` | A `KeyError(None)` reached the UI | Validate input early with messages that name the problem | — (validate in the backend) |

### Known gaps in Synplex (don't copy these)

- **Multi-tab isolation (B5):** some element registries are server-scope. Keep
  all of them in `index()` (REQ-06).
- **Resume can overwrite a run started in another tab (R-G4):** `_offer_resume`
  checks `result.dirs is None` *before* the dialog, not after. Re-check
  `status != "running"` and "nothing loaded" after the await.
- **Redundant rebuilds (R-G8):** a single drag triggers 3 figure rebuilds, and
  some actions rebuild the whole gallery and reset zoom.
- **Non-dict session JSON (R-G9):** add a type check in `load_session`.
- **No mid-job checkpointing:** acceptable for 40 s jobs, not for ML batches
  (§9).

---

## 9. [NEW] Additions for the ML / SEM app

These rules extend the same three-tier model to long batches. None of them
exist in Synplex.

### 9.1 Backend boundary

**REQ-70 — The GUI talks to the backend through a narrow interface.**

```
submit(batch_spec) -> batch_id
status(batch_id)   -> {state, done, total, current_item, log_tail, error}
cancel(batch_id)
results(batch_id)  -> summary (table paths, per-image outputs)
```

The reconcile loop polls `status()` (or reads a server-scope mirror that a
progress callback updates). The same GUI code then works whether the backend
runs in a thread, a subprocess, or a separate service. If the backend runs
outside the GUI process, its manifest on disk is the source of truth, and a
restarted GUI can reattach to a batch that is **still running**.

**REQ-71 — Load the model once, off the event loop.** Load at server start, in
a background task, or lazily on the first job inside the worker. Never load it
inside `index()` or a handler's synchronous body. Show a "model loading" status
through the same reconcile path.

**REQ-72 — The batch spec is the key.** `batch_spec` = input image list (with
content hashes), model id + weights hash, and all inference parameters. If any
of these change, the work is new: start a new batch or invalidate the matching
outputs. Never mix outputs from two specs in one result.

### 9.2 Per-item checkpointing (surviving F7)

**REQ-73 — A manifest per batch, written as work progresses.** Use
`<batch_dir>/manifest.json`, or SQLite for thousands of images. It holds the
spec, plus per image: `pending | done | failed`, the output path, and the error
text. Update it atomically (REQ-54) after each image, or every N images.

**REQ-74 — Each image's output is written before it's marked done.** Write the
result atomically (temp + replace), then update the manifest. A crash between
the two just redoes one image.

**REQ-75 — Resuming a batch skips verified work.** An image counts as done
only if the manifest says so, its output file exists, and its input hash +
model hash + params match the spec. Everything else is re-queued. The
operation is idempotent: resuming twice gives the same result.

**REQ-76 — The Resume prompt shows batch progress.** It reads "Batch *X*:
143 / 500 images done (2 failed). Resume / Discard / Not now". This is the same
offer rules as REQ-58–60. "Resume" re-submits the batch, and the backend skips
done items. A batch whose manifest says `running` but has no live worker is
shown as `interrupted`, not `running`.

**REQ-77 — Cancellation is cooperative and resumable.** A `threading.Event` (or
equivalent) is checked between images. Cancel leaves a clean manifest in
`cancelled` state, which the user can resume later.

**REQ-78 — One bad image doesn't kill the batch.** Catch per-image errors
(decode failure, GPU OOM), mark that image `failed` with the message, and
continue. Show failed images in the UI with a retry action.

**REQ-79 — Partial results are viewable.** While a batch runs, the results area
can show the images and measurements done so far. Gate it on a cheap counter
(e.g. `done` changed by ≥ N or ≥ 5 s elapsed), not every image, so it stays
within REQ-13 and REQ-40.

### 9.3 User edits on images

**REQ-80 — Image annotations are session state.** Scale-bar calibration,
ROI/exclusion regions, accepted/rejected detections and manual corrections are
user edits. They live in server-scope state, autosave through REQ-56, and are
keyed by image content hash, not by list position, so they survive re-ordering
and re-runs of the same images.

**REQ-81 — Derived measurements are recomputed, not persisted.** Fiber
diameters after corrections, histograms and summary stats are rebuilt from the
model output + the user's edits on resume. Don't store a second copy that can
drift.

---

## 10. Verification checklist

Run these in a browser against a real-sized input before trusting the app.
Synplex verified H1–H18 this way. The automated tests cover only the session
round trip (`tests/test_session_store.py`).

1. Start a job, **refresh** mid-run → progress continues live, Run is
   disabled, and results appear on completion without re-running.
2. Start a job, **background the tab** for > 60 s (or sleep the laptop),
   return → same page, same state.
3. Open a **second tab** mid-run → both show progress; neither can start a
   second job.
4. **Close the tab** while a confirm dialog is open → no errors in the
   terminal; nothing destructive happened; the next page behaves normally.
5. **Esc** on the Resume prompt, then refresh → the prompt appears again and
   the session file still exists.
6. Edit annotations/windows, wait 1–2 s, **restart the server**, Resume →
   every edit is back; the safe-mode switch is off.
7. **Delete the output folder**, restart → no Resume prompt (and no error).
8. **Corrupt the session file** (truncate it; replace it with `[]`) and
   restart → no prompt, no crash.
9. Run input A, then input B → none of A's edits or identity show on B. Run
   B with a forced failure → A's results and edits are still consistent, and
   export/record refuses.
10. Measure the websocket payload of one results refresh (browser dev tools →
    WS frames) for the largest realistic input; it must not scale with image
    size or count.
11. **[NEW]** Kill the server mid-batch (Ctrl-C / `kill -9`), restart, Resume
    batch → finished images are skipped; the final results equal those of an
    uninterrupted run.
12. **[NEW]** Change one inference parameter, then Resume → it's treated as a
    new batch, not merged.

---

## Appendix A — Minimal skeleton

This shows how the pieces connect. It isn't a drop-in implementation: error
handling, the results renderer and the backend are stubs.

```python
# state.py — server scope, plain data only (REQ-01, 02, 04)
from dataclasses import dataclass, field

@dataclass
class JobState:
    status: str = "idle"          # idle | running | complete | error | cancelled
    generation: int = 0
    progress: float = 0.0
    done: int = 0
    total: int = 0
    log_lines: list[str] = field(default_factory=list)
    log_total: int = 0
    error: str | None = None
    batch_dir: str | None = None
    source: str | None = None
    summary: object = None        # loaded results; never UI elements
```

```python
# app.py
import asyncio, json
import matplotlib; matplotlib.use("Agg")          # REQ-29, before backend imports
from nicegui import Client, app, ui

STATE = JobState()
EDITS: dict = {}                     # user annotations, server scope
RESUME = {"offered": False}          # REQ-59
LAST_SAVED = {"snap": None}
LOG_MAX = 500

def log(line: str) -> None:          # backend progress callback (REQ-26, 27)
    STATE.log_lines.append(line); STATE.log_total += 1
    del STATE.log_lines[:max(0, len(STATE.log_lines) - LOG_MAX)]

def save_session(only_if_changed=False) -> None:                     # REQ-56
    if not STATE.batch_dir:
        return
    payload = session.build_payload(STATE.batch_dir, STATE.source, EDITS)
    snap = json.dumps(payload, sort_keys=True, default=str)
    if only_if_changed and snap == LAST_SAVED["snap"]:
        return
    if session.save(payload):                                        # atomic, never raises
        LAST_SAVED["snap"] = snap

async def run_job(spec) -> None:     # started by a handler, writes state only
    if STATE.status == "running":                                    # REQ-22
        return
    STATE.status, STATE.error, STATE.progress = "running", None, 0.0
    STATE.log_lines.clear(); STATE.log_total = 0
    try:
        out = await asyncio.to_thread(backend.run_batch, spec, on_log=log)   # REQ-20
        summary = await asyncio.to_thread(backend.load_summary, out)
        if STATE.source != spec.source:                              # REQ-24
            EDITS.clear()
        STATE.batch_dir, STATE.summary, STATE.source = out, summary, spec.source
        STATE.progress = 1.0
        STATE.status = "complete"                                    # REQ-23: flags last
        STATE.generation += 1
        save_session()
    except Exception as exc:
        STATE.error, STATE.status = str(exc), "error"

@ui.page("/")
def index():
    view = {"generation": -1, "status": None, "log_total": 0}       # REQ-05
    status = ui.label()
    bar = ui.linear_progress(value=0)
    run_btn = ui.button("Run", on_click=lambda: run_job(current_spec()))
    log_box = ui.log(max_lines=200)
    results = ui.column()

    def reconcile():                                                 # REQ-12
        try:
            _reconcile()
        except Exception as exc:
            print(f"[gui] reconcile failed: {exc}")

    def _reconcile():
        first = view["status"] is None
        pending = STATE.log_total - view["log_total"]                # REQ-27
        if pending < 0:
            log_box.clear(); pending = STATE.log_total
        if pending:
            for line in STATE.log_lines[-pending:]:
                log_box.push(line)
            view["log_total"] = STATE.log_total
        bar.set_value(STATE.progress)                                # REQ-14
        run_btn.set_enabled(STATE.status != "running")
        if STATE.status != view["status"]:                           # REQ-13
            status.set_text(STATE.error or STATE.status)
        if STATE.generation != view["generation"]:                   # REQ-11
            view["generation"] = STATE.generation
            results.clear()
            with results:
                render_results(STATE.summary, EDITS)                 # lazy tabs, thumbnails
            if not first:
                ui.notify("Analysis finished")                       # REQ-15
        view["status"] = STATE.status
        if STATE.status == "complete":
            save_session(only_if_changed=True)                       # REQ-56

    reconcile()
    ui.timer(0.5, reconcile)                                         # REQ-10

    async def offer_resume():                                        # REQ-57..61
        if STATE.summary is not None or RESUME["offered"]:
            return
        saved = session.load()                                       # validated or None
        RESUME["offered"] = True
        if not saved:
            return
        client = ui.context.client
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Resume {saved['name']}?")
            ui.button("Start fresh", on_click=lambda: dlg.submit("fresh"))
            ui.button("Resume", on_click=lambda: dlg.submit("resume"))
        try:
            answer = await dlg
        except asyncio.CancelledError:
            RESUME["offered"] = False
            raise
        if client.id not in Client.instances or answer is None:      # REQ-30, 31
            RESUME["offered"] = False
            return
        if answer == "fresh":
            session.clear(); return
        if STATE.status == "running" or STATE.summary is not None:   # avoid Synplex R-G4
            return
        summary = await asyncio.to_thread(backend.load_summary, saved["batch_dir"])
        EDITS.clear(); EDITS.update(saved["edits"])                  # REQ-61
        STATE.batch_dir, STATE.source, STATE.summary = saved["batch_dir"], saved["source"], summary
        STATE.progress, STATE.status = 1.0, "complete"
        STATE.generation += 1

    ui.timer(0.3, offer_resume, once=True)

ui.run(reload=False, reconnect_timeout=30.0)                         # REQ-34
```

## Appendix B — Synplex files to read alongside this spec

| File | Read for |
|---|---|
| `src/synplex/gui/state.py` | the whole server-state contract in ~60 lines |
| `src/synplex/session_store.py` | a complete, small restart store (atomic write, versioning, validation, key coercion) |
| `src/synplex/gui/app.py` → `main`, `_PipelineLogStream`, `_save_session`, `handle_run`, `handle_record`, `_page_gone`, `_reconcile_inner`, `_offer_resume`, `ui.run` | every pattern in §2–§5 and §7 |
| `src/synplex/gui/components.py` → `_downsample_lttb`, `plot_gallery` (lazy tabs), `_render_tracking_panel` (per-item refresh, drag select), `_zoomable_image` (cache bust), `file_config_tile` | §6 |
| `docs/archive/STABILITY_UX_PLAN.md` | the original diagnosis with measurements (H1–H6) |
| `CHANGELOG.md` entries 2026-09-19 → 2026-10-22 | how each fix landed |
| `tests/test_session_store.py` | the session round-trip tests to port |
