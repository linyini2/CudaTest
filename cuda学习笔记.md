1、在使用11-stream中，kernel-transfer.cu创建多个cuda流，实现核函数执行和数据传输的并行，测试发现，创建的cuda流最大在76000左右（超过2的16次方65536），不能超过76500（会报错：out of memory）。cuda流个数在16左右时运行时间最少。

![image-20220305161203535](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220305161203535.png)

加速比分别为：1.67，1.66



2、![image-20220305163430424](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220305163430424.png)

一个线程块最多的线程数是1024，也就是三个方向乘积必须小于等于1024

最大的线程数是4亿7千万左右，

![image-20220308161457888](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220308161457888.png)



![image-20220308193004227](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220308193004227.png)

test-shared.cu：使用了共享内存，可以调用的最大线程数在14亿左右。



![image-20220308202755926](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220308202755926.png)

test-totalshared.cu：测试每个block可以使用的共享内存的大小，结果是47.5K/per block，跟系统指定的48K/per block差不多。总共使用的共享内存是18G，若将BLOCK_SIZE增加至79，会报错（如下图）。

![image-20220308204714594](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220308204714594.png)

