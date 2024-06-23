from flask import Flask, render_template, request
import numpy as np

# create 5 XYZRGB points
xs = np.asarray([0,1,2,3,4,5,6,7,8])
ys = np.asarray([0,1,2,3,4,5,6,7,8])
zs = np.asarray([0,1,2,3,4,5,6,7,8])
reds = np.asarray([255,235,215,195,175])
greens = np.asarray([0,0,0,0,0])
blues = np.asarray([0,0,0,0,0])

app = Flask(__name__)

@app.route('/some_place')
def hello_world():
	some_variable_name = request.args.get('some_variable_name')
	print(some_variable_name)
	return 'Hello, World!'


@app.route('/other_place')
def other_world():
	return render_template(	'hello.html', 
							hello_name="3co",
							another_thing="blah",
							n_size = 13,
							json_variable={"thing": 1})


@app.route('/render_3d_data')
def other_2_world():
	# may be doing some actual point processing here
	# actually set up my final point variable data here
	return render_template(	'render_3d_points.html', 
							xs=xs,
							ys=ys,
							zs=zs,
							reds=reds,
							blues=blues,
							greens=greens)

