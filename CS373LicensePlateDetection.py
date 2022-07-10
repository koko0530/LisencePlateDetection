import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from queue import Queue
# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# rgb图片转为灰度图
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            greyscale_pixel_array[y][x] = round(
        pixel_array_r[y][x] * 0.299 + pixel_array_g[y][x] * 0.587 + pixel_array_b[y][x] * 0.114)
    return greyscale_pixel_array


# 使用5x5标准差滤波器
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    new_arr = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(2, image_height - 2, ):
        for y in range(2, image_width - 2):
            avg = (pixel_array[x - 1][y + 1] + pixel_array[x][y + 1] + pixel_array[x + 1][y + 1] + pixel_array[x - 1][
                y] + pixel_array[x][y] + pixel_array[x + 1][y] + pixel_array[x - 1][y - 1] + pixel_array[x][y - 1] +
                   pixel_array[x + 1][y - 1] + pixel_array[x - 2][y + 2] + pixel_array[x - 1][y + 2] + pixel_array[x][
                       y + 2] + pixel_array[x + 1][y + 2] + pixel_array[x + 2][y + 2] + pixel_array[x - 2][y + 1] +
                   pixel_array[x + 2][y + 1] + pixel_array[x - 2][y] + pixel_array[x + 2][y] + pixel_array[x - 2][
                       y - 1] + pixel_array[x + 2][y - 1] + pixel_array[x - 2][y - 2] + pixel_array[x - 1][y - 2] +
                   pixel_array[x][y - 2] + pixel_array[x + 1][y - 2] + pixel_array[x + 2][y - 2]) / 25
            new_arr[x][y] = math.sqrt((pow((pixel_array[x - 1][y + 1] - avg), 2) + pow((pixel_array[x][y + 1] - avg),
                                                                                       2) + pow(
                (pixel_array[x + 1][y + 1] - avg), 2) + pow((pixel_array[x - 1][y] - avg), 2) + pow(
                (pixel_array[x][y] - avg), 2) + pow((pixel_array[x + 1][y] - avg), 2) + pow(
                (pixel_array[x - 1][y - 1] - avg), 2) + pow((pixel_array[x][y - 1] - avg), 2) + pow(
                (pixel_array[x + 1][y - 1] - avg), 2) + pow((pixel_array[x - 2][y + 2] - avg), 2) + pow(
                (pixel_array[x - 1][y + 2] - avg), 2) + pow((pixel_array[x][y + 2] - avg), 2) + pow(
                (pixel_array[x + 1][y + 2] - avg), 2) + pow((pixel_array[x + 2][y + 2] - avg), 2) + pow(
                (pixel_array[x + 2][y + 1] - avg), 2) + pow((pixel_array[x + 2][y + 1] - avg), 2) + pow(
                (pixel_array[x - 2][y] - avg), 2) + pow((pixel_array[x + 2][y] - avg), 2) + pow(
                (pixel_array[x - 2][y - 1] - avg), 2) + pow((pixel_array[x + 2][y - 1] - avg), 2) + pow(
                (pixel_array[x - 2][y - 2] - avg), 2) + pow((pixel_array[x - 1][y - 2] - avg), 2) + pow(
                (pixel_array[x + 1][y - 2] - avg), 2) + pow((pixel_array[x][y - 2] - avg), 2) + pow(
                (pixel_array[x + 1][y - 2] - avg), 2) + pow((pixel_array[x + 2][y - 2] - avg), 2)) / 25)
    return new_arr


# 线性拉缩至0-255
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    max_v = 0
    min_v = 255
    for y in range(0, image_height):
        for x in range(0, image_width):
            if pixel_array[y][x] > max_v:
                max_v = pixel_array[y][x]
            elif pixel_array[y][x] < min_v:
                min_v = pixel_array[y][x]

    if image_height == image_width:
        pixel_array = [[0, 0], [0, 0]]
    else:
        for x in range(0, image_height):
            for y in range(0, image_width):
                pixel_array[x][y] = round((pixel_array[x][y] - min_v) * ((255 / (max_v - min_v))))
    return pixel_array

# 使用阈值将图二值化
threshold_value = 120


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for x in range(0, image_height):
        for y in range(0, image_width):
            if pixel_array[x][y] < threshold_value:
                pixel_array[x][y] = 0
            elif pixel_array[x][y] >= threshold_value:
                pixel_array[x][y] = 255
    return pixel_array


# 扩张
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    new_arr = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(1, image_height - 1):
        for y in range(1, image_width - 1):
            if pixel_array[x][y] != 0:
                new_arr[x][y] = 1
                new_arr[x - 1][y - 1] = 1
                new_arr[x + 1][y + 1] = 1
                new_arr[x][y - 1] = 1
                new_arr[x + 1][y - 1] = 1
                new_arr[x - 1][y] = 1
                new_arr[x + 1][y] = 1
                new_arr[x][y + 1] = 1
                new_arr[x - 1][y + 1] = 1
            elif pixel_array[0][0] == 255:
                new_arr[0][0] = 1
                new_arr[0][1] = 1
                new_arr[1][0] = 1
                new_arr[1][1] = 1
    return new_arr


# 腐蚀
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    new_arr = createInitializedGreyscalePixelArray(image_width, image_height);
    for x in range(1, image_height - 1):
        for y in range(1, image_width - 1):
            if pixel_array[x - 1][y - 1] == 0 or pixel_array[x][y - 1] == 0 or pixel_array[x + 1][y - 1] == 0 or \
                    pixel_array[x - 1][y] == 0 or pixel_array[x + 1][y] == 0 or pixel_array[x][y] == 0 or \
                    pixel_array[x - 1][y + 1] == 0 or pixel_array[x][y + 1] == 0 or pixel_array[x + 1][y + 1] == 0:
                new_arr[x][y] = 0
            else:
                new_arr[x][y] = 1
    return new_arr

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)

    current_label = 1
    pixel_size = {}

    for x in range(image_height):
        for y in range(image_width):
            if (pixel_array[x][y] > 0) and (visited[x][y] == 0):
                q = Queue()
                q.put((y, x))
                visited[x][y] = 1
                total_pixel = 0
                while q.empty() == False:
                    (y1, x1) = q.get()
                    result[x1][y1] = current_label
                    total_pixel += 1
                    if (y1 - 1 >= 0) and (pixel_array[x1][y1 - 1] > 0) and (visited[x1][y1 - 1] == 0):
                        q.put((y1 - 1, x1))
                        visited[x1][y1 - 1] = 1

                    if (y1 + 1 <= image_width - 1) and (pixel_array[x1][y1 + 1] > 0) and (visited[x1][y1 + 1] == 0):
                        q.put((y1 + 1, x1))
                        visited[x1][y1 + 1] = 1

                    if (x1 - 1 >= 0) and (pixel_array[x1 - 1][y1] > 0) and (visited[x1 - 1][y1] == 0):
                        q.put((y1, x1 - 1))
                        visited[x1 - 1][y1] = 1

                    if (x1 + 1 <= image_height - 1) and (pixel_array[x1 + 1][y1] > 0) and (
                            visited[x1 + 1][y1] == 0):
                        q.put((y1, x1 + 1))
                        visited[x1 + 1][y1] = 1

                pixel_size[current_label] = total_pixel
                current_label += 1
    return (result, pixel_size)


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')
    # STUDENT IMPLEMENTATION here
    # 将RGB转换为grey
    grey = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    # 拉伸至0到255
    grey_s = scaleTo0And255AndQuantize(grey, image_width, image_height)
    # 找到高对比区域
    contrast = computeStandardDeviationImage5x5(grey_s, image_width, image_height)
    # 拉伸至0到255
    contrast_s = scaleTo0And255AndQuantize(contrast, image_width, image_height)
    # 二值化
    contrast_b = computeThresholdGE(contrast_s, threshold_value, image_width, image_height)
    # 膨胀
    contrast_d = computeDilation8Nbh3x3FlatSE(contrast_b, image_width, image_height)
    # 腐蚀
    contrast_e = computeErosion8Nbh3x3FlatSE(contrast_d, image_width, image_height)
    # connect
    connect, sizes = computeConnectedComponentLabeling(contrast_e, image_width, image_height)
    # 根据面积对sizes的key进行排序
    sizes_list = sorted(sizes.items(), key = lambda kv:(kv[1], kv[0]))
    target_labels = sizes_list[::-1]
    #这里从最大区域开始遍历，判断连通域的形状长宽比是否满足要求，如果不满足则遍历第二大的连通域，以此类推
    for i, (target_label, _) in enumerate(target_labels):
        # bbox范围
        bbox_min_x = image_width
        bbox_max_x = 0
        bbox_min_y = image_height
        bbox_max_y = 0
        for y in range(image_height):
            for x in range(image_width):
                if connect[y][x] == target_label: # 找到对应label的像素点
                    if y > bbox_max_y:
                        bbox_max_y = y
                    if y < bbox_min_y:
                        bbox_min_y = y
                    if x > bbox_max_x:
                        bbox_max_x = x
                    if x < bbox_min_x:
                        bbox_min_x = x
        # bbox的长宽
        bbox_height = bbox_max_y - bbox_min_y
        bbox_width = bbox_max_x - bbox_min_x
        if bbox_width / bbox_height < 8: # 如果长宽比大于一定比例，说明识别错误，没识别到车牌，则遍历下一个面积次大的区域
            # print(i)
            break
    px_array = px_array_r

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()