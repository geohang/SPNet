%% FORWARD MODELING CODE FOR SP SIGNAL GENERATION BY FINITE-INFINITE ELEMENT COUPLING METHOD
%% Model Pre-definition
NX = 10;                                                    % Number of Elements in X-axis
NY = 10;                                                    % Number of Elements in Y-axis
NZ = 10;                                                    % Number of Elements in Z-axis
X  = -5:5;                                                  % Element Subdivision in X-axis (Has to be Symmetric about the Origin)
Y  = -5:5;                                                  % Element Subdivision in X-axis (Has to be Symmetric about the Origin)
Z  = 0:-1:-10;                                              % Element Subdivision in X-axis (Has to decrease from the Origin)
NE = NX*NY*NZ+NX*NZ*2+NY*NZ*2+NX*NY;                        % Total Number of Elements
ND = (NX+1)*(NY+1)*(NZ+1)+2*(NX+NY)*(NZ+1)+(NX-1)*(NY-1);   % Total Number of Nodes

%% Gauss Integral Node (3,4,and 6,respectively) for Infinite Elements
Coordinate=[-0.7745967 0 0.7745967];
Weight=[0.5555556 0.8888889 0.5555556];
% Coordinate=[-0.8611363 -0.33998104 0.33998104 0.8611363];
% Weight=[0.3478548 0.6521451 0.6521451 0.3478548];
% Coordinate=[-0.9324695 -0.6612094 -0.2386192 0.2386192 0.6612094 -0.9324695];
% Weight=[0.1713245 0.3607616 0.4679139 0.4679139 0.3607616 0.1713245];

%% Electric Conductivity Distribution
sigma_background=1/100;      % Unit (S/m)
SGM(1:NE)=sigma_background;  % User-defined

%% 计算节点信息
[XYZ,I8]=Subfunction_XYZI8(X,Y,Z,NX,NY,NZ,NE,ND);

%% 有限元区域六面体单元分析
Kfinite=Subfunction_Finite_Hexahedron(SGM,I8,XYZ,NX,NY,NZ,ND);                                  % Coefficient Matrix from Finite Elements

%% 无限元区域单元分析
Kinfinite=Subfunction_Infinite_Element(sigma_background,I8,XYZ,Coordinate,Weight,NX,NY,NZ,ND);  % Coefficient Matrix from Infinite Elements

%% 总体合成
Kfinite=Kinfinite+Kfinite;   % Total Coefficient Matrix

%% Source Vector
Source=zeros(ND,1);          % Source Vector
Source_location=100;         % Source Location Found through Node Number 
Source(Source_location)=-1;  % Unit (mA)

%% Solve the System of Partial Differential Equations by 'BiConjugate Gradient Stabilized based on Incomplete LU Matrix Decomposition'
u=zeros(ND,1);                                                                              % SP Vector
Source=Source-Kfinite*u;
freeNode=1:ND;
tol=1e-8;                                                                                   % tolerance
maxit=10000;                                                                                % Maximum Iterations
[LTE,UTE]=ilu(Kfinite(freeNode,freeNode));                                                  % Incomplete LU Matrix Decomposition
[u(freeNode),~,~]=bicgstab(Kfinite(freeNode,freeNode),Source(freeNode),tol,maxit,LTE,UTE);  % BiConjugate Gradient Stabilized