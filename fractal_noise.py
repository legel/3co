import numpy as np
from PIL import Image

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
        
def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.9):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise
    
if __name__ == '__main__': 
    image_size = 2048

    all_reds = []
    all_greens = []
    all_blues = []

    np.random.seed(0)
    for resolution_size in [128, 128, 2, 4, 8, 16, 32, 64, 64, 64]:
        print("resolution {}".format(resolution_size))
        all_reds.append(generate_fractal_noise_2d((image_size, image_size), (resolution_size, resolution_size), 5))
        all_greens.append(generate_fractal_noise_2d((image_size, image_size), (resolution_size, resolution_size), 5))
        all_blues.append(generate_fractal_noise_2d((image_size, image_size), (resolution_size, resolution_size), 5))

    img = Image.new('RGB', (image_size, image_size), color = 'black')
    pixels = img.load()

    unnormed_reds = []
    unnormed_greens = []
    unnormed_blues = []

    for h in range(image_size):
        for v in range(image_size):

            red = int(255 * sum([(reds[h][v] + 1.0 / 2.0) for reds in all_reds]) / len(all_reds))
            green = int(255 * sum([(greens[h][v] + 1.0 / 2.0) for greens in all_greens]) / len(all_greens))
            blue = int(255 * sum([(blues[h][v] + 1.0 / 2.0) for blues in all_blues]) / len(all_blues))

            unnormed_reds.append(red)
            unnormed_greens.append(green)
            unnormed_blues.append(blue)

    min_red = np.percentile(unnormed_reds, 5)
    max_red = np.percentile(unnormed_reds, 95)
    min_green = np.percentile(unnormed_greens, 5)
    max_green = np.percentile(unnormed_greens, 95)
    min_blue = np.percentile(unnormed_blues, 5)   
    max_blue = np.percentile(unnormed_blues, 95)

    for h in range(image_size):
        for v in range(image_size):
            red = int(255 * (unnormed_reds[h*image_size + v] - min_red) / (max_red - min_red))
            green = int(255 * (unnormed_greens[h*image_size + v] - min_green) / (max_green - min_green))
            blue = int(255 * (unnormed_blues[h*image_size + v] - min_blue) / (max_blue - min_blue))

            pixels[h,v] = (red, green, blue, 1)

        print("({},{}): R:{}, G:{}, B:{}".format(h,v, red, green, blue))

    img.save('normed_hierarchical_perlin_noise.png')
