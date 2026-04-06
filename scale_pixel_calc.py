import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os

class ScaleCalibrator:
    def __init__(self, root):
        self.root = root
        self.root.title("SEM Scale Calibrator")
        
        self.image_path = None
        self.points = [] # Original image coordinates
        self.orig_image = None
        self.tk_image = None
        self.zoom_level = 1.0
        self.zoom_step = 1.2
        
        # UI Setup
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_open = tk.Button(self.toolbar, text="Open Image", command=self.load_image)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_zoom_in = tk.Button(self.toolbar, text="Zoom In (+)", command=lambda: self.zoom(1.2))
        self.btn_zoom_in.pack(side=tk.LEFT, padx=2)
        
        self.btn_zoom_out = tk.Button(self.toolbar, text="Zoom Out (-)", command=lambda: self.zoom(0.8))
        self.btn_zoom_out.pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = tk.Label(self.toolbar, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = tk.Button(self.toolbar, text="Reset Points", command=self.reset_points)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.info_label = tk.Label(self.toolbar, text="Select two points on the X-axis")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Scrollable Canvas
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Control-plus>", lambda e: self.zoom(1.2))
        self.root.bind("<Control-minus>", lambda e: self.zoom(0.8))
        
        # Initial status
        self.status = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if not file_path:
            return
            
        self.image_path = file_path
        self.orig_image = Image.open(file_path)
        self.zoom_level = 1.0
        
        # If image is too big, start with a reasonable zoom
        screen_h = self.root.winfo_screenheight() - 200
        if self.orig_image.height > screen_h:
            self.zoom_level = screen_h / self.orig_image.height
            
        self.update_view()
        self.reset_points()
        self.status.config(text=f"Loaded: {os.path.basename(file_path)} ({self.orig_image.width}x{self.orig_image.height})")

    def zoom(self, factor):
        if not self.orig_image: return
        self.zoom_level *= factor
        # Constrain zoom
        self.zoom_level = max(0.1, min(self.zoom_level, 10.0))
        self.update_view()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom(1.1)
        else:
            self.zoom(0.9)

    def update_view(self):
        if not self.orig_image: return
        
        new_size = (int(self.orig_image.width * self.zoom_level), int(self.orig_image.height * self.zoom_level))
        
        # Use NEAREST for speed and pixel clarity when zoomed in
        resample_mode = Image.NEAREST if self.zoom_level > 1 else Image.LANCZOS
        resized = self.orig_image.resize(new_size, resample_mode)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))
        
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.redraw_markers()

    def reset_points(self):
        self.points = []
        self.canvas.delete("marker")
        self.canvas.delete("line")
        self.info_label.config(text="Click the first point of the distance.")

    def redraw_markers(self):
        self.canvas.delete("marker")
        self.canvas.delete("line")
        r = 3
        scaled_points = []
        for p in self.points:
            sx, sy = p[0] * self.zoom_level, p[1] * self.zoom_level
            scaled_points.append((sx, sy))
            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill="red", outline="white", tags="marker")
        
        if len(scaled_points) == 2:
            self.canvas.create_line(scaled_points[0][0], scaled_points[0][1], 
                                   scaled_points[1][0], scaled_points[1][1], 
                                   fill="yellow", width=2, tags="line")

    def on_click(self, event):
        if not self.orig_image:
            return
            
        # Get coordinates relative to current zoom
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        # Map back to original image pixels
        ox = cx / self.zoom_level
        oy = cy / self.zoom_level
        
        self.points.append((ox, oy))
        
        if len(self.points) == 1:
            self.info_label.config(text="Click the second point.")
            self.redraw_markers()
        elif len(self.points) == 2:
            self.redraw_markers()
            self.calculate()
        else:
            self.reset_points()
            self.on_click(event)

    def calculate(self):
        p1, p2 = self.points
        
        dx = abs(p2[0] - p1[0])
        
        self.info_label.config(text=f"X distance: {dx:.2f} pixels. Waiting for input...")
        
        um_val = simpledialog.askfloat("Input", f"Measured X distance: {dx:.2f} pixels.\nEnter the physical distance in micrometers (um):", parent=self.root)
        
        if um_val is not None and um_val > 0:
            pixels_per_um = dx / um_val
            result_msg = (
                f"Selected X distance: {dx:.2f} pixels\n"
                f"Physical distance: {um_val} um\n\n"
                f"RESULT: {pixels_per_um:.4f} pixels/um\n"
                f"(Inverse: {1/pixels_per_um:.6f} um/pixel)"
            )
            messagebox.showinfo("Scale Result", result_msg)
            print(f"--- Calibration for {os.path.basename(self.image_path)} ---")
            print(result_msg)
            self.status.config(text=f"Last Calibration: {pixels_per_um:.4f} px/um")
        
        self.info_label.config(text="Click to start over or Open new image.")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = ScaleCalibrator(root)
    root.mainloop()
