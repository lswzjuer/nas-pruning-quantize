
def histgram(name,values,path,index=None,type=None,sqnr=None):
    if not os.path.exists(path):
        os.mkdir(path)
    picname = name + "_{}".format(index) if index is not None else name
    picnamepath=os.path.join(path,picname+".png")
    # 
    values=values.flatten()
    # from collections import Counter
    # print(Counter(list(values)))
    # set the plt
    fig=plt.figure()
    plt.grid()
    tname=name + " ({})".format(type) if type is not None else name
    stname= tname + "{}(db)".format(sqnr) if sqnr is not None else tname
    plt.title(stname)
    plt.xlabel('value')
    plt.ylabel('frequency')
    #plt.ylim(0,4)
    #plt.xlim(-5,5)
    plt.hist(values,bins="auto",density=True, histtype='bar', facecolor='blue')
    fig.savefig(picnamepath, bbox_inches='tight')
    plt.close(fig)