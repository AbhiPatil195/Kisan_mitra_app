import os
import uvicorn
import sys

def start_server():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the correct directory
    os.chdir(current_dir)
    
    # Add the current directory to Python path
    sys.path.insert(0, current_dir)
    
    # Start the server with proper configuration
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable auto-reload to prevent loops
        log_level="info",
        workers=1,  # Use single worker
        loop="auto",
        timeout_keep_alive=30,
        access_log=True,
        use_colors=True
    )

if __name__ == "__main__":
    start_server() 