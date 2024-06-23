from robotics import Iris

iris = Iris()
# iris.initialize_robot()
# iris.focus_optics(distance=0.55)
iris.scan(auto_focus=False, auto_exposure=False, hdr_exposure_times=[5.0, 60.0, 120.0, 200.0, 250.0, 300.0, 350.0, 400.0], save_exr=True)

# iris.initialize_focus()
#iris.scan(auto_focus=False, auto_exposure=False, distance=0.3, exposure_time=)