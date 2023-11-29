# coding=utf-8
from setuptools import setup

setup(
    author="zata",
    description="This is a nir analyse api, writen by zata",   ### 一句话概括一下
    name="nirapi",   ### 给你的包取一个名字
    version="1.0",   ### 你的包的版本号
    packages=['API_FOR_NIR_ANALYSIS-master'], ### 这里写的是需要从哪个文件夹下导入python包，如果找不到会报错，默认你下载下来解压之后的文件夹名就是API_FOR_NIR_ANALYSIS-master
    exclude_package_date={'':['.gitignore'], '':['dist'], '':'build', '':'utility.egg.info'},    ### 这是需要排除的文件，也就是只把有用的python文件导入到环境变量中

)
