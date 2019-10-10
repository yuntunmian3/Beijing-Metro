import sys
import io
import numpy as np

sys.stdout=io.TextIOWrapper(sys.stdout.detach(),encoding='utf-8')
sys.stdin=io.TextIOWrapper(sys.stdin.detach(),encoding='utf-8')
LINEDATA=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','24']
lineName={
    '1':'1号线',
    '2':'2号线',
    '3':'4号线',
    '4':'5号线',
    '5':'6号线',
    '6':'7号线',
    '7':'8号线南',
    '8':'8号线北',
    '9':'9号线',
    '10':'10号线',
    '11':'13号线',
    '12':'14号线东',
    '13':'14号线西',
    '14':'15号线',
    '15':'16号线',
    '16':'S1号线',
    '17':'S2号线',
    '18':'八通线',
    '19':'昌平线',
    '20':'房山线',
    '21':'首都机场线',
    '22':'西郊线',
    '23':'燕房线',
    '24':'亦庄线'
}
STATION_NUM={}
data={}
datanum={}
with open("stations.txt",'r',encoding = 'utf-8') as f:
    TOTAL=f.readline()
    for line in f.readlines():
        if line!='\n':
            line = line.rstrip('\n')
            line=line.split(' ')
            if line[0] in LINEDATA:
                linei=line[0]
                continue
            # print(line)
            line[1]=linei
            line0=line[0]
            intline=int(line[1])
            if intline not in data.keys():
                data[intline]=[line0]
            else:
                data[intline].append(line0)
            if line0 not in datanum.keys():
                datanum[line0]=[intline]
            else:
                datanum[line0].append(intline)
i=0
STATIO={}
for datai in datanum.keys():
    STATION_NUM[datai]=i
    STATIO[i]=datai
    i+=1
# print(STATION_NUM)#对应到邻接矩阵的引索上
I=i
#判断是否为环线
def iscircle(mlist):
    if mlist[0]==mlist[-1]:
        return True
    return False

#判断是否为换乘点
def istransport(station):
    if len(datanum[station])>1:
        return True
    return False
#得到换线路
def destransport(station):
    return datanum[station]
# print(datanum)
def changeline(p1,p2):
    line1=datanum[p1]
    line2=datanum[p2]
    a=[]
    # print(data[line1[0]])
    for i1 in data[line1[0]]:
        if istransport(i1):
            ways=destransport(i1)
            for i2 in line2:
                if i2 in ways:
                    a.append(i1)
                    return i1
    return None
class GraphError(ValueError):
    pass
class Graph:
    def __init__(self,mat,unconn=0):
        vnum=len(mat)
        for x in mat:
            if len(x)!=vnum:#检查是否是方阵
                raise ValueError("Argument for 'Graph'.")
        self._mat=[mat[i][:] for i in range(vnum)]#赋值mat到self._mat
        self._unconn=unconn
        self._vnum=vnum
    def vertex_num(self):
        return self._vnum
    def _invalid(self,v):
        return 0>v or v>=self._vnum
    def add_vertex(self):#并未计划支持增加顶点，所以直接定义为错误，要增加顶点需要增加一行矩阵一列
        raise GraphError("Adj-Matrix does not support 'add_vertex'.")
    def add_edge(self,vi,vj,val=1):
        if self._invalid(vi) or self._invalid(vj):
            raise GraphError(str(vi)+' or '+str(vj)+" is not a valid vertex.")
        self._mat[vi][vj]=val
    def get_edge(self,vi,vj):
        if self._invalid(vi) or self._invalid(vj):
            raise GraphError(str(vi)+' or '+str(vj)+" is not a valid vertex.")
        return self._mat[vi][vj]
   #记录已经构造的表
    #用静态方法构造结点表
    def out_edges(self,vi):
        if self._invalid(vi):
            raise GraphError(str(vi)+" is not a valid vertex.")
        return self._out_edges(self._mat[vi],self._unconn)
    @staticmethod
    def _out_edges(row,unconn):
        edges=[]
        for i in range(len(row)):
            if row[i]!=unconn:
                edges.append((i,row[i]))
            return edges
class GraphAL(Graph):
    def __init__(self,mat=[],unconn=0):
        vnum=len(mat)
        for x in mat:
            if len(x)!=vnum:
                raise ValueError("Argument for 'GraphAL'.")
        self._mat=[Graph._out_edges(mat[i],unconn) for i in range(vnum)]
        self._vnum=vnum
        self._unconn=unconn
    def add_vertex(self):#增加新节点时安排一个新编号
        self._mat.append([])
        self._vnum+=1
        return self._vnum-1
    def add_edge(self,vi,vj,val=1):
        if self._vnum==0:
            raise GraphError("cannot add edge to empty graph")
        if self._invalid(vi) or self._invalid(vj):
            raise GraphError(str(vi) + ' or ' + str(vj) + " is not a valid vertex.")
        row=self._mat[vi]
        i=0
        while i<len(row):
            if row[i][0]==vj:#更新mat[vi][vj]的值
                self._mat[vi][i]=(vj,val)
                return
            if row[i][0]>vj:#原来如果没有到vj的边，退出循环，加入边
                break
            i+=1
        self._mat[vi].insert(i,(vj,val))
    def get_edge(self,vi,vj):
        if self._invalid(vi) or self._invalid(vj):
            raise GraphError(str(vi) + ' or ' + str(vj) + " is not a valid vertex.")
        for i,val in self._mat[vi]:
            if i==vj:
                return val
        return self._unconn
    def out_edges(self,vi):
        if self._invalid(vi):
            raise GraphError(str(vi)+" is not a valid vertex.")
        return self._mat[vi]

mat=np.zeros([i,i])
mat=np.full([i,i],np.inf)
RouteGraph=Graph(mat)
routee={}
for key in data.keys():
    datai=data[key]
    # print(datai)
    for i in range(1,len(datai)-1):
        # RouteGraph.add_vertex()
        v1=STATION_NUM[datai[i]]
        v2=STATION_NUM[datai[i+1]]
        v3=STATION_NUM[datai[i-1]]
        RouteGraph.add_edge(v1, v2, 1)
        RouteGraph.add_edge(v2, v1, 1)
        RouteGraph.add_edge(v3, v1, 1)
        RouteGraph.add_edge(v1, v3, 1)
    if iscircle(datai):
        # RouteGraph.add_vertex()
        v1=STATION_NUM[datai[0]]
        v2=STATION_NUM[datai[-2]]
        RouteGraph.add_edge(v1, v2, 1)
        RouteGraph.add_edge(v2, v1, 1)


def all_shortest_path(graph):
    import numpy as np
    vnum=graph.vertex_num()
    a=[[graph.get_edge(i,j) for j in range(vnum)]for i in range(vnum)]
    nvertex=[[-1 if a[i][j]==np.inf else j for j in range(vnum)]for i in range(vnum)]
    for k in range(vnum):
        for i in range(vnum):
            for j in range(vnum):
                if a[i][j]>a[i][k]+a[k][j]:
                    a[i][j] = a[i][k] + a[k][j]
                    nvertex[i][j]=nvertex[i][k]
    return(a,nvertex)
def find_shortest_path(graph, start, end, path=[]):
    '查找最短路径'
    path = path + [start]
    if start == end:
        return path
    if not start in graph.keys():
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest
def find_all_paths(graph, start, end, path):
    '查找所有的路径'
    path = path + [start]
    if start == end:
        return [path]
    if not start in graph.keys():
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

class PrioQueueError(ValueError):
    pass
#基于堆的优先队列类,在尾端加入元素，首端作为堆顶，见peek等
class PrioQueue:
    def __init__(self,elist=[]):
        self._elems=list(elist)
        if elist:
            self.buildheap()
    def buildheap(self):
        end=len(self._elems)
        for i in range(end//2,-1,-1):
            self.siftdown(self._elems[i],i,end)
    def is_empty(self):
        return not self._elems
    def peek(self):
        if self.is_empty():
            raise PrioQueueError("in peek")
        return self._elems[0]
    def enqueue(self,e):
        self._elems.append(None)
        self.siftup(e,len(self._elems)-1)
    def siftup(self,e,last):
        elems,i,j=self._elems,last,(last-1)//2
        while i>0 and e<elems[j]:
            elems[i]=elems[j]
            i,j=j,(j-1)//2
        elems[i]=e
    def dequeue(self):
        if self.is_empty():
            raise PrioQueueError("in dequeue")
        elems=self._elems
        e0=elems[0]
        e=elems.pop()
        if len(elems)>0:
            self.siftdown(e,0,len(elems))
        return e0
    def siftdown(self,e,begin,end):
        elems,i,j=self._elems,begin,begin*2+1
        while j<end:
            if j+1<end and elems[j+1]<elems[j]:
                j+=1
            if e<elems[j]:
                break
            elems[i]=elems[j]
            i,j=j,2*j+1
        elems[i]=e
#Dijkstra算法实现最短路径查找
pathss={}
for i in range(I):
    for j in range(I):
        if RouteGraph.get_edge(i,j)==1:
            start=STATIO[i]
            end=STATIO[j]
            if i not in pathss.keys():
                pathss[i]=[j]
            else:
                pathss[i].append(j)
print(pathss)
def dijkstra_shortest_pathS(graph,v0,endpos):
    vnum=0
    for i in pathss.keys():
        vnum+=1
    # print(vnum)
    # vnum=graph.vertex_num()
    assert 0<=v0<vnum
    paths=[None]*vnum#长为vnum的表记录路径
    count=0
    cands=PrioQueue([(0,v0,v0)])#求解最短路径的候选边集记录在优先队列cands中（p,v,v'）v0经过v到v'的最短路径长度为p，根据p的大小排序，保证选到最近的未知距离顶点
    while count<vnum and not cands.is_empty():
        plen,u,vmin=cands.dequeue()#取路径最短顶点
        # print(u,vmin)
        if paths[vmin]:#如果这个点的最短路径已知，则跳过
            continue
        paths[vmin]=(u,plen)#新确定最短路径并记录
        for v in graph[vmin]:#遍历经过新顶点组的路径
            if not paths[v]:#如果还不知道最短路径的顶点的路径，则记录
                cands.enqueue((plen+1,vmin,v))
        count+=1
        # print(paths)
    return paths

def getPath(startpos,endpos):
    s1=STATION_NUM[startpos]
    e1=STATION_NUM[endpos]
    # print(s1,e1)
    paths=dijkstra_shortest_pathS(pathss,s1,e1)
    # print(paths)
    b=[]
    p=paths[e1][0]
    b.append(STATIO[p])
    while True:
        p1=paths[p][0]
        p=p1
        b.append(STATIO[p])
        if p==s1:
            break
    b.reverse()
    # print(b)
    if len(datanum[b[0]])==1:
        lines=datanum[b[0]][0]
    else:
        for i in datanum[b[0]]:
            for j in datanum[b[1]]:
                if i==j:
                    lines=i
    # print(lines)
    ways=[]
    ways.append([b[0],lines])
    # print(STATION_NUM)
    for i in range(len(b)-1):
        li=datanum[b[i]]
        if len(li)>1:
            for j in li:
                if j!=lines and j in datanum[b[i+1]]:
                    lines=j
                    ways.append([b[i],lines])
    ways.append([STATIO[e1]])
    result=''
    stations=[]
    route=[]
    if startpos == endpos:
        result = str(startpos)+'---'+str(lineName[str(ways[i][1])])+'(0站)'+'---'+str(endpos)
        return result
    for i in range(len(ways)-1):
        length=paths[STATION_NUM[ways[i+1][0]]][1]-paths[STATION_NUM[ways[i][0]]][1]
        result = result + str(ways[i][0])   #站名
        result = result + '---'
        result = result + str(lineName[str(ways[i][1])])   #线路名
        result = result + '('
        result = result + str(length) + '站'      #乘坐长度
        result = result + ')'
        if i!=len(ways)-1:
            result = result + '---'
    result = result + str(ways[-1][0])

    return result
