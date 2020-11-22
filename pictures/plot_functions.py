# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-12 16:34:10
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-19 00:00:00


import numpy as np 
import matplotlib.pyplot as plt


# ndarray
def sign(dlist):
    dlist[dlist<0] = -1
    dlist[dlist>=0] = 1
    return dlist

def signGrade(dlist):
    mask1 = dlist<0.002
    mask2 = dlist>-0.002
    mask = mask1*mask2
    dlist = dlist*0
    dlist[mask]=100
    return dlist


def htanh(dlist):
    dlist[dlist<=-1] = -1
    dlist[dlist>=1] = 1
    return dlist

def htanhGrade(dlist):
    mask1 = dlist>=-1
    mask2 = dlist<=1
    mask = mask1*mask2
    grade = np.zeros_like(dlist)
    grade[mask] = 1
    return grade

def approx(dlist):
    mask1 = dlist < -1
    mask2 = dlist < 0
    mask3 = dlist < 1
    mask = -1 * mask1.astype(np.float32) + (dlist * 2 + np.power(dlist,2)) * (1 - mask1.astype(np.float32))
    mask = mask * mask2.astype(np.float32) + (dlist * 2 - np.power(dlist,2)) * (1 - mask2.astype(np.float32))
    mask = mask * mask3.astype(np.float32) + 1 * (1 - mask3.astype(np.float32))
    return mask


def approxGrade(dlist):
    mask1 = dlist < -1
    mask2 = dlist < 0
    mask3 = dlist < 1
    mask = 0 * mask1.astype(np.float32) + (dlist * 2 + 2) * (1 - mask1.astype(np.float32))
    mask = mask * mask2.astype(np.float32) + (-dlist * 2 + 2) * (1 - mask2.astype(np.float32))
    mask = mask * mask3.astype(np.float32) + 0 * (1 - mask3.astype(np.float32))
    return mask



def tanhpx(dlist,alpha):

    px=np.exp(alpha*dlist)
    fpx=np.exp(-alpha*dlist)
    res = (px-fpx)/(px+fpx)
    return res


def tanhpxGrade(dlist,alpha):
    res = alpha * (1-np.power(tanhpx(dlist,alpha),2))
    return res






if __name__ == '__main__':


    testlist1 = np.linspace(start=-2,stop=0,num=1000)
    testlist2 = np.linspace(start=0,stop=2,num=1000)
    flist = list(testlist1)+list(testlist2)[1:]
    flist = np.asarray(flist)


    sign_list = sign(flist.copy())
    sign_grad = signGrade(flist.copy())

    htanh_list = htanh(flist.copy())
    htanh_grad = htanhGrade(flist.copy())


    approx_list = approx(flist.copy())
    approx_grad=approxGrade(flist.copy())

    tanhpx_list1 = tanhpx(flist.copy(),alpha=1)
    tanhpx_list2 = tanhpx(flist.copy(),alpha=2)
    tanhpx_list3 = tanhpx(flist.copy(),alpha=3)

    tanhpx_grade1 = tanhpxGrade(flist.copy(),alpha=1)
    tanhpx_grade2 = tanhpxGrade(flist.copy(),alpha=2)
    tanhpx_grade3 = tanhpxGrade(flist.copy(),alpha=3)


    bluelist=['lavender',"lightsteelblue",'cornflowerblue','royalblue','blue','mediumblue','darkblue','navy']
    fig = plt.figure()

    # add subplot1
    sub1 = fig.add_subplot(1, 2, 1)
    sub1.set_title("Forward")
    sub1.plot(flist,sign_list,color = 'black', linewidth = 2.0, linestyle = '-',
              label=r"$sign(.)$")
    sub1.plot(flist,htanh_list,color = 'red', linewidth = 2.0, linestyle = '-',
              label=r"$Htanh(.)$")
    sub1.plot(flist,approx_list,color = 'orange', linewidth = 2.0, linestyle = '-',
              label=r"$ApproxFunc(.)$")

    sub1.fill_between(flist, sign_list, htanh_list, color="g", alpha=0.3)
    sub1.fill_between(flist, sign_list, approx_list, color="b", alpha=0.3)


    # sub1.plot(flist,tanhpx_list1,color = bluelist[0], linewidth = 2.0, linestyle = '-',
    #           label=r"$tanhx(.)$")
    # sub1.plot(flist,tanhpx_list2,color = bluelist[1], linewidth = 2.0, linestyle = '-',
    #           label=r"$tanh2x(.)$")
    # sub1.plot(flist,tanhpx_list3,color = bluelist[2], linewidth = 2.0, linestyle = '-',
    #           label=r"$tanh3x(.)$")



    sub1.set_xlim(-2, 2)
    sub1.set_ylim(-3, 3)
    sub1.set_xlabel('x')
    sub1.set_ylabel('$F(x)$')
    sub1.grid(linestyle='-.')
    handles, labels = sub1.get_legend_handles_labels()
    # reverse the order
    sub1.legend(loc=4)


    sub2 = fig.add_subplot(1, 2, 2)
    sub2.set_title("Backward")
    sub2.plot(flist,sign_grad,color = 'black', linewidth = 2.0, linestyle = '-',
              label=r"$\frac{\mathrm{d}sign(x)}{\mathrm{d}x}$")
    sub2.plot(flist,htanh_grad,color = 'red', linewidth = 2.0, linestyle = '-',
              label=r"$\frac{\mathrm{d}Htanh(x)}{\mathrm{d}x}$")
    sub2.plot(flist,approx_grad,color = 'orange', linewidth = 2.0, linestyle = '-',
              label=r"$\frac{\mathrm{d}ApproxFunc(x)}{\mathrm{d}x}$")

    sub2.set_xlim(-2, 2,0.5)
    sub2.set_ylim(-3, 3,0.5)
    sub2.set_xlabel('x')
    sub2.set_ylabel(r'$\frac{\mathrm{d}F(x)}{\mathrm{d}x}$')
    sub2.grid(linestyle='-.')
    sub2.legend(loc=4)

    plt.savefig("./sign_htanh_approx.png")
    plt.close()
