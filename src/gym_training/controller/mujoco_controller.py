class UR5e_controller():
    def __init__(self, model=None, data=None, render=None):
        self.model = model
        self.data = data
        self.render = render

    
    def move_ee_carthesian(x, y, z, r, p, y):
        
    
    def get_image_data(self, width, height):
        rgb, depth = self.render(offwidth=width, offheight=height)
        print("hallo")