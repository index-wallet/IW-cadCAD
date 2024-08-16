## built-in
import multiprocessing

## custom
from ui.webgui import prepare_and_get_dash_app

if __name__ == "__main__":

    ## Main guard is REQUIRED for multiprocessing to work in Windows

    ## We use multiprocessing in the webgui to speed up the centrality layout generation
    ## However since this 'technically' isn't the main process, we need to call freeze_support
    multiprocessing.freeze_support()

    app = prepare_and_get_dash_app(is_debug=False)
    
    app.run(port="8050")