# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-12-15 22:56:37
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-12-16 23:39:37



import numpy as np 
from collections import OrderedDict


VGGDICT={
    # input image: h w c , k,k,o
    "conv0":[32,32,3,3,3,128],
    "conv1":[32,32,128,3,3,128],

    "conv2":[16,16,128,3,3,256],
    "conv3":[16,16,256,3,3,256],

    "conv4":[8,8,256,3,3,512],
    "conv5":[8,8,512,3,3,512],

    "fc":[4,4,512,10]
}


def ger_params():

    params=OrderedDict()
    params_count =0
    flops_count = 0
    b_coun = 0
    f_count = 0
    for name,v in VGGDICT.items():
        if name!="fc":
            h,w,cin,k,cout = v[0],v[1],v[2],v[3],v[5]
            if name == "conv0":
                Bflops = 0
                flops = k*k*cin*h*w*cout
            else:
                Bflops = k*k*cin*h*w*cout
                flops = h*w*cout
            pa = k*k*cin*cout
            params[name] = [Bflops,flops,pa]
            params_count += pa
            b_coun +=Bflops
            f_count += flops
            print(name,Bflops/1e6,flops/1e6,pa/1e3)
        else:
            h,w,cin,cout = v[0],v[1],v[2],v[3]
            Bflops = h*w*cout*cin
            flops = h*w*cout*cin
            pa = h*w*cout*cin
            params[name] = [Bflops,flops,pa]
            params_count += pa
            b_coun +=Bflops
            f_count += flops
            print(name,Bflops/1e6,flops/1e6,pa/1e3)

    print(b_coun/1e6,f_count/1e6,params_count/1e3)
if __name__ == '__main__':
    ger_params()
