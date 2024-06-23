def simplest_cb(image_name, image_data, percent=1):
    out_channels = []
    cumstops = (
        image_data.shape[0] * image_data.shape[1] * percent / 200.0,
        image_data.shape[0] * image_data.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv.split(image_data):
        cumhist = np.cumsum(cv.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv.LUT(channel, lut.astype('uint8')))
    merged = cv.merge(out_channels)
    cv.imwrite(image_name, merged) 

def normalize_colors(image_name):
    image_data = cv.imread(image_name)
    simplest_cb(image_name=image_name, image_data=image_data)