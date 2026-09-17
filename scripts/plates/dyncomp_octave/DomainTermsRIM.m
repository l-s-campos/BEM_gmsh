function [M2,res2]=DomainTermsRIM(GEO_MESH,PHY_NODES,GEO_NODES, ...
    INTERNAL_POINTS,aprfun,dA,rho)

global hpl q NPGSF NPGID TOL rho hpl


fprintf('\n3. CONSTRUYENDO MATRIZ [M] Y VECTOR [Q] \n')
fprintf('\t3.1 Construyendo vector [Q] \n')

n_noc = length(PHY_NODES(:,1));
if(isempty(INTERNAL_POINTS))
    n_noi=0;
else
    n_noi=length(INTERNAL_POINTS(:,1)); % Number of internal nodes
end


n_nodes = n_noc + n_noi;

drm_nodes = PHY_NODES(:,1:3);
source_points = PHY_NODES(:,1:3);

if(n_noi~=0)
    drm_nodes(n_noc+1:n_noc+n_noi,:) = INTERNAL_POINTS;
    source_points(n_noc+1:n_noc+n_noi,:) = INTERNAL_POINTS;
end

n_drm_nodes=length(drm_nodes(:,1));
%Estas dos lineas fueron comentadas porque la numeraci�n de los nodos de
%internos empieza desde el final de los nodos de cortorno.
%drm_nodes(n_noc+1:n_drm_nodes,1) = (n_noc+1:n_drm_nodes)';
%source_points(n_noc+1:n_nodes,1) = (n_noc+1:n_nodes)';
%-------------------------------------------------------------------------

if(aprfun==4||aprfun==3)
    m=n_drm_nodes+3;
    m2=n_drm_nodes+2;
else
    m=n_drm_nodes;
    m2=m;
end
% C�lculo da matriz [F]
for i = 1 : n_drm_nodes
    xi = drm_nodes(i,2);
    yi = drm_nodes(i,3);
    iF=3*i-2:3*i;
    iI=3;
    domain_load = [0 0 q];
    body(iF) = domain_load;
    PHI(3*i-2:3*i,3*i-2:3*i) = [rho*hpl^3/12     0                0;...
                                    0         rho*hpl^3/12        0;...
                                    0            0              rho*hpl];
    for j = 1 : m2
        if(j<=n_noc+n_noi)
            jF=3*j-2:3*j;
            jI=3;
        else
            if(i<=n_noc+n_noi)
                jF=3*(n_noc+n_noi)+ ...
                    3*(j-((n_noc+n_noi)))-2;
                jI=3;
            else
                jF=3*(n_noc+n_noi)+ ...
                    3*(j-((n_noc+n_noi)));
                jI=1;
            end;
        end;
        if(iI==jI)
            Id=eye(length(iF));
        elseif(iI>jI)
            Id=flipud(eye(length(iF),length(jF)));
        else
            Id=fliplr(eye(length(iF),length(jF)));
        end
        if(j<=n_drm_nodes)
            xj = drm_nodes(j,2);
            yj = drm_nodes(j,3);
            r = sqrt((xi-xj)^2+(yi-yj)^2);
            if(aprfun==1)
                F(iF,jF)=Id*(1-r)*Id;
            elseif(aprfun==2)
                F(iF,jF)=(1+r+r^3)*Id;
            elseif(aprfun==3)
                if(r>0)
                    F(iF,jF)=r^2*log(r)*Id;
                else
                    F(iF,jF)=zeros(iI,jI);
                end
            elseif(aprfun==4)
                if(r<dA)
                    F(iF,jF)=(1-6*(r/dA)^2+8*(r/dA)^3-3*(r/dA)^4)*Id;
                else
                    F(iF,jF)=zeros(iI,jI);
                end;
            end;
        elseif(j==n_drm_nodes+1)
            F(iF,jF:jF+2)=xi*Id;
        else
            F(iF,jF:jF+2)=yi*Id;
        end
    end
end

if(aprfun==4||aprfun==3)
    n_lines=length(F(:,1));
    n_columns=length(F(1,:));
    F(n_lines+1:n_columns,1:n_lines)=F(1:n_lines,n_lines+1:n_columns)';
    F(n_columns+1:n_columns+3,1:3*(n_noc+n_noi))= ...
        repmat(eye(3),1,n_noc+n_noi);
    F(n_columns+1:n_columns+3,3*(n_noc+n_noi)+1: ...
        3*(n_noc+n_noi)+9)=repmat(eye(3),1,3);
    F(1:3*(n_noc+n_noi)+9,n_columns+1:n_columns+3)= ...
        F(n_columns+1:n_columns+3,1:3*(n_noc+n_noi)+9)';
end
if(aprfun==4||aprfun==3)
    alpha = inv(F) * [body zeros(1,9)]';
else
    alpha = inv(F) * body';
end;
bd=F*alpha;

n_el = length(GEO_MESH(:,1));	% N. total de elementos

if(aprfun==4||aprfun==3)
    M=zeros(3*n_noc+3*n_noi,3*(n_noc+n_noi)+9);
else
    M=zeros(3*n_noc+3*n_noi,3*(n_noc+n_noi));
end

number_of_GaussPoint1=4;
[Gauss_p1,Gauss_w1]=Gauss_Legendre(-1,1,number_of_GaussPoint1);

number_of_GaussPoint2=6;
[Gauss_p2,Gauss_w2]=Gauss_Legendre(-1,1,number_of_GaussPoint2);

Gauss_p1=Gauss_p1';
Gauss_w1=Gauss_w1';

Gauss_p2=Gauss_p2';
Gauss_w2=Gauss_w2';
%--------------------------------------------------------------------------

fprintf('\t3.2 Construyendo matriz [M] \n')

for noRD = 1 : m
    Avance=noRD*100/m;
    fprintf('Avance. %g  \n',Avance);
    jM=3*noRD-2;
    if(noRD<=n_noc+n_noi)
        xr = drm_nodes(noRD,2);
        yr = drm_nodes(noRD,3);
        augm = 0;
    elseif(noRD==n_nodes+1)
        augm = 1;
        xr = 0;
        yr = 0;
    elseif(noRD==n_nodes+2)
        augm = 2;
        xr = 0;
        yr = 0;
    else
        augm = 3;
        xr = 0;
        yr = 0;
    end
    for i_sourcep=1:n_nodes
        x_f=source_points(i_sourcep,2);
        y_f=source_points(i_sourcep,3);
        IM=3*i_sourcep-2;        
        for el = 1 : n_el
            % Numera��o dos tr�s n�s do elemento i
            node1_GEO = GEO_MESH(el,2);
            node3_GEO = GEO_MESH(el,4);

            x1 = GEO_NODES(node1_GEO,2);	y1 = GEO_NODES(node1_GEO,3);
            x2 = GEO_NODES(node3_GEO,2);	y2 = GEO_NODES(node3_GEO,3);

            [m_el] = compute_surf_int(x_f,y_f,x1,y1,x2,y2,...
                xr,yr,aprfun,augm,...
                dA,Gauss_p1,Gauss_w1,Gauss_p2,Gauss_w2);
            M(IM:IM+2,jM:jM+2)=M(IM:IM+2,jM:jM+2)+m_el;
        end
    end
end

res2=M*alpha;
M = M*inv(F);
n_col=3*(n_noi+n_noc); 
M2=M(:,1:n_col)*PHI;
