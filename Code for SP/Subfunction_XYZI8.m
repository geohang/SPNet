%% 节点信息子函数
function [XYZ,I8]=Subfunction_XYZI8(X,Y,Z,NX,NY,NZ,NE,ND)
%本函数为区域长方体网格剖分函数，计算节点坐标及节点编号；从右列到左列，从上行到下行，从前面到后面；
%参数说明：XYZ为三维数组，存放各节点的xyz坐标、I8为二维数组，存放长方体单元节点编号；
%          坐标系：x轴向右，y轴向里，z轴向上，原点在地表中心；
multiple = 3.5;
XYZ = zeros(ND,3);
I8 = zeros(NE,8);
nd = (NX+1)*(NY+1)*(NZ+1);

%% 节点初始坐标
for i=1:NX+1
    for j=1:NZ+1
        for k=1:NY+1
            N=(i-1)*(NY+1)*(NZ+1)+(j-1)*(NY+1)+k;
            XYZ(N,1)=X(i);
            XYZ(N,2)=Y(k);
            XYZ(N,3)=Z(j);
        end
    end
end

%% 无限单元节点坐标
%X正向（前侧）的单一映射无限单元的坐标定义（行右到行左，上行到下行）
k=1;
for i=nd+1:nd+(NY+1)*(NZ+1)
    XYZ(i,1)=multiple*XYZ(k,1);
    XYZ(i,2)=multiple*XYZ(k,2);
    XYZ(i,3)=multiple*XYZ(k,3);
    k=k+1;
end

%X负向（后侧）的单一映射无限单元的坐标定义（行右到行左，上行到下行）
k=NX*(NY+1)*(NZ+1)+1;
for i=nd+(NY+1)*(NZ+1)+1:nd+(NY+1)*(NZ+1)*2
    XYZ(i,1)=multiple*XYZ(k,1);
    XYZ(i,2)=multiple*XYZ(k,2);
    XYZ(i,3)=multiple*XYZ(k,3);
    k=k+1;
end

%Y正向（右侧）的单一映射无限单元的坐标定义（列上到列下，外列到里列）
k_right=zeros(1,(NX-1)*(NZ+1));
for i=1:NX-1
    for j=1:NZ+1
        k_right(j+(i-1)*(NZ+1))=(NY+1)*(NZ+1)+1+(j-1)*(NY+1)+(i-1)*(NY+1)*(NZ+1);
    end
end
k=1;
for i=nd+(NY+1)*(NZ+1)*2+1:nd+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)
    XYZ(i,1)=multiple*XYZ(k_right(k),1);
    XYZ(i,2)=multiple*XYZ(k_right(k),2);
    XYZ(i,3)=multiple*XYZ(k_right(k),3);
    k=k+1;
end

%Y负向（左侧）的单一映射无限单元的坐标定义（列上到列下，外列到里列）
k_left=zeros(1,(NX-1)*(NZ+1));
for i=1:NX-1
    for j=1:NZ+1
        k_left(j+(i-1)*(NZ+1))=(NY+1)*(NZ+1)+NY+1+(j-1)*(NY+1)+(i-1)*(NY+1)*(NZ+1);
    end
end
k=1;
for i=nd+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+1:nd+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2
    XYZ(i,1)=multiple*XYZ(k_left(k),1);
    XYZ(i,2)=multiple*XYZ(k_left(k),2);
    XYZ(i,3)=multiple*XYZ(k_left(k),3);
    k=k+1;
end

%Z负向（下侧）的单一映射无限单元的坐标定义（行右到行左，外行到里行）
k_down=zeros(1,(NX-1)*(NY-1));
for i=1:NX-1
    for j=1:NY-1
        k_down(j+(i-1)*(NY-1))=(NY+1)*(NZ+1)+(NY+1)*NZ+2+(j-1)+(i-1)*(NY+1)*(NZ+1);
    end
end
k=1;
for i=nd+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+1:nd+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+(NX-1)*(NY-1)
    XYZ(i,1)=multiple*XYZ(k_down(k),1);
    XYZ(i,2)=multiple*XYZ(k_down(k),2);
    XYZ(i,3)=multiple*XYZ(k_down(k),3);
    k=k+1;
end

%% 添加高程
% D=Z(NZ-3);
% Zp=[0 0 1 1.8 2.8 4 5.5 6.2 7.1 7.8 ...
%     8.2 8.3 7 6.5 5 3.5 2.2 1.3 0.8 0.1 ...
%     -0.3 -0.6 -1 -1.6 -2 -2.3 -2.8 -3.3 -3.6 -4.1 ...
%     -4.2 -4.2 -3.9 -3.6 -3.4 -3.5 -3.2 -2.7 -2 -1.4 ...
%     -0.5 0.3 1 1.2 1.5 1.6 1.9 2.6 2.8 3 ...
%     4 4.1 4 3.5 3 2.5 1.5 0.8 0.3 0 ...
%     0];
% count=1;
% for ii=1:NX+1
% for i=6:NY-4
%     for j=1:NZ-1
%         XYZ((NY+1)*(NZ+1)*(ii-1)+i+(j-1)*(NY+1),3)=abs(D-Z(j))*Zp(count)/abs(D)+XYZ((NY+1)*(NZ+1)*(ii-1)+i+(j-1)*(NY+1),3); 
%     end
% end
% count=count+1;
% end
 
%% 单元编号
%% 有限元单元编号
kk=0;kkk=0;
for i=1:NX
    for j=1:NZ
        for k=1:NY
            N=(i-1)*NY*NZ+(j-1)*NY+k;     
            I8(N,8)=(i-1)*(NY+1)*(NZ+1)+(j-1)*(NY+1)+k;
            I8(N,7)=I8(N,8)+1;
            I8(N,6)=I8(N,7)+(NY+1)*(NZ+1);
            I8(N,5)=I8(N,8)+(NY+1)*(NZ+1);
            I8(N,4)=I8(N,8)+(NY+1);
            I8(N,3)=I8(N,4)+1;
            I8(N,2)=I8(N,3)+(NY+1)*(NZ+1);
            I8(N,1)=I8(N,4)+(NY+1)*(NZ+1);
        end
        kk=kk+1;
    end
    kkk=kkk+1;
end

%% 无限元单元编号
%X正向（前侧）的单一映射无限单元的节点编号（行右到行左，上行到下行）i=(NX+1)*(NY+1)*(NZ+1)+1:(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)
for j=1:NZ
    for k=1:NY
        N=NX*NY*NZ+(j-1)*NY+k;
        I8(N,8)=nd+k+(j-1)*(NY+1);
        I8(N,7)=I8(N,8)+1;
        I8(N,6)=1+k+(j-1)*(NY+1);
        I8(N,5)=I8(N,6)-1;
        I8(N,4)=I8(N,8)+(NY+1);
        I8(N,3)=I8(N,4)+1;
        I8(N,2)=I8(N,6)+(NY+1);
        I8(N,1)=I8(N,2)-1;
    end
end

%X负向（后侧）的单一映射无限单元的坐标定义（行右到行左，上行到下行）i=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+1:(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2
for j=1:NZ
    for k=1:NY
        N=NX*NY*NZ+NY*NZ+(j-1)*NY+k;
        I8(N,5)=nd+(NY+1)*(NZ+1)+k+(j-1)*(NY+1);
        I8(N,6)=I8(N,5)+1;
        I8(N,1)=I8(N,5)+NY+1;
        I8(N,2)=I8(N,1)+1;
        I8(N,8)=(NY+1)*(NZ+1)*NX+k+(j-1)*(NY+1);
        I8(N,7)=I8(N,8)+1;
        I8(N,4)=I8(N,8)+(NY+1);
        I8(N,3)=I8(N,4)+1;
    end
end

%Y正向（右侧）的单一映射无限单元的坐标定义（列上到列下，外列到里列）i=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+1:(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)
for j=1:NX-1
    for k=1:NZ
        N=NX*NY*NZ+NY*NZ*2+(j-1)*NZ+k;
        I8(N,5)=nd+(NY+1)*(NZ+1)*2+k+(j-1)*(NZ+1);
        I8(N,1)=I8(N,5)+1;
        I8(N,6)=(NY+1)*(NZ+1)+1+(k-1)*(NY+1)+(j-1)*(NY+1)*(NZ+1);
        I8(N,2)=I8(N,6)+NY+1;
        I8(N,7)=1+(k-1)*(NY+1)+(j-1)*(NY+1)*(NZ+1);
        I8(N,3)=I8(N,7)+NY+1;
    end
end
for k=1:NZ
    N=NX*NY*NZ+NY*NZ*2+(NX-1)*NZ+k;
    I8(N,5)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+1+(k-1)*(NY+1);
    I8(N,1)=I8(N,5)+NY+1;
    I8(N,6)=(NY+1)*(NZ+1)*NX+1+(k-1)*(NY+1);
    I8(N,2)=I8(N,6)+NY+1;
    I8(N,7)=I8(N,6)-(NY+1)*(NZ+1);
    I8(N,3)=I8(N,7)+NY+1;
end
for k=1:NZ
    N=NX*NY*NZ+NY*NZ*2+k;
    I8(N,8)=(NX+1)*(NY+1)*(NZ+1)+1+(k-1)*(NY+1);
    I8(N,4)=I8(N,8)+NY+1;
end
for j=2:NX
    for k=1:NZ
        N=NX*NY*NZ+NY*NZ*2+(j-1)*NZ+k;
        I8(N,8)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+k+(j-2)*(NZ+1);
        I8(N,4)=I8(N,8)+1;
    end
end

%Y负向（左侧）的单一映射无限单元的坐标定义（列上到列下，外列到里列）i=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+1:(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2
for j=1:NX-1
    for k=1:NZ
        N=NX*NY*NZ+NY*NZ*2+NX*NZ+(j-1)*NZ+k;
        I8(N,6)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+k+(j-1)*(NZ+1);
        I8(N,2)=I8(N,6)+1;
        I8(N,5)=(NY+1)*(NZ+1)+NY+1+(k-1)*(NY+1)+(j-1)*(NY+1)*(NZ+1);
        I8(N,1)=I8(N,5)+NY+1;
        I8(N,8)=NY+1+(k-1)*(NY+1)+(j-1)*(NY+1)*(NZ+1);
        I8(N,4)=I8(N,8)+NY+1;
    end
end
for k=1:NZ
    N=NX*NY*NZ+NY*NZ*2+NX*NZ+(NX-1)*NZ+k;
    I8(N,6)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+NY+1+(k-1)*(NZ+1);
    I8(N,2)=I8(N,6)+NY+1;
    I8(N,5)=(NY+1)*(NZ+1)*NX+NY+1+(k-1)*(NY+1);
    I8(N,1)=I8(N,5)+NY+1;
    I8(N,8)=I8(N,5)-(NY+1)*(NZ+1);
    I8(N,4)=I8(N,8)+NY+1;
end
for k=1:NZ
    N=NX*NY*NZ+NY*NZ*2+NX*NZ+k;
    I8(N,7)=(NX+1)*(NY+1)*(NZ+1)+NY+1+(k-1)*(NY+1);
    I8(N,3)=I8(N,7)+NY+1;
end
for j=2:NX
    for k=1:NZ
        N=NX*NY*NZ+NY*NZ*2+NX*NZ+(j-1)*NZ+k;
        I8(N,7)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+k+(j-2)*(NZ+1);
        I8(N,3)=I8(N,7)+1;
    end
end

%Z负向（下侧）的单一映射无限单元的坐标定义（行右到行左，外行到里行）i=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+1:(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+(NX-1)*(NY-1)
for j=1:NX
    for k=1:NY
        N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+(j-1)*NY+k;
        I8(N,8)=(NY+1)*NZ+k+(j-1)*(NY+1)*(NZ+1);
        I8(N,7)=I8(N,8)+1;
        I8(N,6)=I8(N,7)+(NY+1)*(NZ+1);
        I8(N,5)=I8(N,6)-1;
    end
end
N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+1;
I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*NZ+1;
I8(N,3)=I8(N,4)+1;
I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+NZ+1;
I8(N,2)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+1;
for k=2:NY-1
    N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+k;
    I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*NZ+k;
    I8(N,3)=I8(N,4)+1;
    I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+k-1;
    I8(N,2)=I8(N,1)+1;
end
N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+NY;
I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*NZ+NY;
I8(N,3)=I8(N,4)+1;
I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+NY-1;
I8(N,2)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+NZ+1;
for j=2:NX-1
    N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+1+(j-1)*NY;
    I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+NZ+1+(j-2)*(NZ+1);
    I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+NZ+1+(j-1)*(NZ+1);
    I8(N,3)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+1+(j-2)*(NY-1);
    I8(N,2)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+1+(j-1)*(NY-1);
    for k=2:NY-1
        N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+k+(j-1)*NY;
        I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+k-1+(j-2)*(NY-1);
        I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+k-1+(j-1)*(NY-1);
        I8(N,3)=I8(N,4)+1;
        I8(N,2)=I8(N,1)+1;
    end
    N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+NY+(j-1)*NY;
    I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+NY-1+(j-2)*(NY-1);
    I8(N,3)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+NZ+1+(j-2)*(NZ+1);
    I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+NY-1+(j-1)*(NY-1);
    I8(N,2)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)+NZ+1+(j-1)*(NZ+1);
end
N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+NY*(NX-1)+1;
I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1);
I8(N,3)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+(NX-2)*(NY-1)+1;
I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+(NY+1)*NZ+1;
I8(N,2)=I8(N,1)+1;
for k=2:NY-1
    N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+NY*(NX-1)+k;
    I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+(NX-2)*(NY-1)+k-1;
    I8(N,3)=I8(N,4)+1;
    I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+(NY+1)*NZ+k;
    I8(N,2)=I8(N,1)+1;
end
N=NX*NY*NZ+NY*NZ*2+NX*NZ*2+NY*NX;
I8(N,4)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2+(NX-1)*(NY-1);
I8(N,3)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)*2+(NX-1)*(NZ+1)*2;
I8(N,1)=(NX+1)*(NY+1)*(NZ+1)+(NY+1)*(NZ+1)+(NY+1)*NZ+NY;
I8(N,2)=I8(N,1)+1;
end