**Stop button to add**
- [ ] fixed
- 
**Run .venv/Scripts/python.exe main.py --debug**
- [x] disables needing model weights

**Top basic/advanced toggle throws error on press:**
- [x] Traceback (most recent call last):
  File "C:\Users\Brend\Documents\GitHub\SEM-Annotator\.venv\Lib\site-packages\nicegui\events.py", line 455, in handle_event
    result = cast(Callable[[EventT], Any], handler)(arguments)
  File "C:\Users\Brend\Documents\GitHub\SEM-Annotator\ui\header.py", line 29, in _on_mode
    state.set_advanced_mode(e.value == 'Advanced')
                            ^^^^^^^
AttributeError: 'GenericEventArguments' object has no attribute 'value'

**All images are .tifs!**
 - [x] wont load, should be fixed.

**too many images were being loaded simultaneously**
 - [x] fixed

**could not clear set scale**
- [x] fixed
- [x] validate with imagej


