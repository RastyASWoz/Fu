import numpy as np
import matplotlib.pyplot as plt
import image
import cal1

image_path = 'img/3.jpeg'

# 画出边界
def plot_curve(curve_points):
    plt.figure(figsize=(6,6))
    plt.plot(curve_points[:,0],curve_points[:,1])
    plt.show()
    # for i in range(0,len(curve_points),50):
    #     plt.scatter(curve_points[i][0],curve_points[i][1],c='r')
    #     plt.draw()
    #     plt.pause(0.1)

# 画出傅里叶级数的结果
def plot_res1( curve_points,n,mode=0):
    thetas = cal1.get_thetas(curve_points)
    a_ns,r_0 = cal1.get_a_ns(thetas,curve_points,n)
    
    # 生成一个θ的值的序列，例如从0到2π，步长为0.01
    drawthetas = np.arange(0,2* np.pi, 0.01)

    # 使用matplotlib的plot函数绘制图像
    if(mode == 0):
        plt.title('n = ' + str(n))
        f_values = [cal1.F(r_0,theta, a_ns, n) for theta in drawthetas]
        f_values = np.array(f_values)
        plt.plot(f_values[:,0],f_values[:,1])
        plt.plot(curve_points[:,0],curve_points[:,1])
        plt.figure(figsize=(6,8))
        #plt.savefig('out/41f'+str(n)+'.jpg')
        plt.show()
    # else是为了展示当n增大时的变化
    else:
        for i in range(2,n) :
            plt.title('n = ' + str(i))
            f_values = [cal1.F(r_0,theta, a_ns, i) for theta in drawthetas]
            f_values = np.array(f_values)
            plt.plot(f_values[:,0],f_values[:,1])
            plt.plot(curve_points[:,0],curve_points[:,1])
            plt.draw()
            plt.pause(0.1)
            plt.clf()

curve_points = image.get_curve_points(image_path)
plot_curve(curve_points)
curve_points = np.array(curve_points)
plot_res1(curve_points,50,1)
