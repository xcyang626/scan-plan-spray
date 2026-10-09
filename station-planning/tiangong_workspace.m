%% 天工3.0 右臂喷涂 - 固定站位可达区(排查对称性来源)
% 原因: 此前将 waist_yaw 在URDF全范围[-150,185]°采样, 腰一转右肩绕中心扫弧,
%       使右臂可够到左右两侧 -> 区域被"对称化". 本脚本对比不同腰部扭转范围.
% 结论: 面向墙面喷涂的合理扭转范围(±30°)下, 右臂可达区明显偏向右(负y)侧.
clear; clc; close all;

WALL_X      = 0.80;
WRIST_PLANE = 0.40;
FAN_ANGLE   = 60;
N_PER       = 800000;   % 每种扭转范围采样数

lim.waist_roll  = deg2rad([-25, 25]);
lim.waist_pitch = deg2rad([-30, 50]);
lim.shpitch     = deg2rad([-163, 163]);
lim.shroll      = deg2rad([-191, 11]);     % 右臂
lim.shyaw       = deg2rad([-163, 163]);
lim.elbpitch    = deg2rad([-146, 11]);
pz_range = [0.40, 0.96];
px_range = [0.00, 0.10];

% 腰部yaw扭转范围(度): 名称
cases = { '无扭转[-5,5]', deg2rad([-5,5]); ...
          '合理扭转[-30,30]', deg2rad([-30,30]); ...
          '大扭转[-90,90]', deg2rad([-90,90]); ...
          'URDF全范围[-150,185]', deg2rad([-150,185]) };
nc = size(cases,1);

res = cell(nc,1);
for ci = 1:nc
    yawr = cases{ci,2};
    rng(2026+ci);
    q1 = yawr(1)            + diff(yawr)            .* rand(N_PER,1);   % 腰yaw
    q2 = lim.waist_roll(1)  + diff(lim.waist_roll)  .* rand(N_PER,1);
    q3 = lim.waist_pitch(1) + diff(lim.waist_pitch) .* rand(N_PER,1);
    q4 = lim.shpitch(1)     + diff(lim.shpitch)     .* rand(N_PER,1);
    q5 = lim.shroll(1)      + diff(lim.shroll)      .* rand(N_PER,1);
    q6 = lim.shyaw(1)       + diff(lim.shyaw)       .* rand(N_PER,1);
    q7 = lim.elbpitch(1)    + diff(lim.elbpitch)    .* rand(N_PER,1);
    px = px_range(1) + diff(px_range) .* rand(N_PER,1);
    pz = pz_range(1) + diff(pz_range) .* rand(N_PER,1);
    wx=zeros(N_PER,1);wy=zeros(N_PER,1);wz=zeros(N_PER,1);
    for i=1:N_PER
        [wx(i),wy(i),wz(i)] = TiangongArmFK_right(px(i),pz(i),[q1(i) q2(i) q3(i) q4(i) q5(i) q6(i) q7(i)]);
    end
    inPlane = abs(wx-WRIST_PLANE)<0.02;
    Y=wy(inPlane); Z=wz(inPlane);
    % 面积(网格)
    dy=0.02;dz=0.02; ymin=-1.0;ymax=1.0;zmin=0;zmax=2.2;
    ny=round((ymax-ymin)/dy)+1; nz=round((zmax-zmin)/dz)+1;
    reg=false(nz,ny);
    for i=1:numel(Y)
        c1=round((Y(i)-ymin)/dy)+1; r1=round((Z(i)-zmin)/dz)+1;
        if r1>=1&&r1<=nz&&c1>=1&&c1<=ny, reg(r1,c1)=true; end
    end
    area_v = nnz(reg)*dy*dz;
    res{ci} = struct('Y',Y,'Z',Z,'area',area_v,'y0',min(Y),'y1',max(Y),'z0',min(Z),'z1',max(Z));
    fprintf('%-20s 采样%d  y:%.2f~%.2f(宽%.2f) z:%.2f~%.2f(高%.2f) 面积%.3f m^2 中心y=%.2f\n', ...
        cases{ci,1}, numel(Y), min(Y),max(Y),range(Y),min(Z),max(Z),range(Z),area_v, mean([min(Y) max(Y)]));
end

%% 绘图: 右臂可达区(不同扭转范围) - 清晰版
colors = {[0.90 0.50 0.05],[0.85 0.15 0.15],[0.15 0.45 0.85],[0.45 0.45 0.50]};
figure('Position',[80 80 860 580],'Color','w'); hold on; box on; grid on; axis equal;
% 先画所有填充(半透明, 从大到小以便内部可见)
for ci = nc:-1:1
    Y=res{ci}.Y; Z=res{ci}.Z;
    k=boundary(Y,Z,0.4);
    fill(Y(k),Z(k),colors{ci},'FaceAlpha',0.22,'EdgeColor','none');
end
% 再画所有边界(置于顶层, 保证每个范围边界清晰可见)
for ci = 1:nc
    Y=res{ci}.Y; Z=res{ci}.Z;
    k=boundary(Y,Z,0.4);
    plot(Y(k),Z(k),'Color',colors{ci},'LineWidth',2.2);
end
plot([0 0],[0 2.2],'--','Color',[0.4 0.4 0.4]);
legend(cases{1,1},cases{2,1},cases{3,1},cases{4,1},'Location','northeast');
text(0.03,2.15,'机器人中心线 y=0','FontSize',9);
xlabel('墙面横向 y (m)   (负=机器人右侧)'); ylabel('墙面高度 z (m)');
title('右臂腕部可达区(40cm平面) - 不同腰部扭转范围对比(含面积)');
% 面积标注(错开位置避免重叠)
for ci = 1:nc
    Y=res{ci}.Y; Z=res{ci}.Z;
    switch ci
        case 1, tx=min(Y)+0.02; ty=max(Z);   ha='left';
        case 2, tx=mean(Y);     ty=0.02;     ha='center';
        case 3, tx=mean(Y);     ty=2.05;     ha='center';
        case 4, tx=max(Y)-0.02; ty=max(Z);   ha='right';
    end
    text(tx,ty,sprintf('%s: %.2f m^2',cases{ci,1},res{ci}.area), ...
        'Color',colors{ci},'FontWeight','bold','FontSize',9,'HorizontalAlignment',ha);
end
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','tiangong_twist_compare.png'));

%% 主图: 合理扭转(±30°)下的右臂可达区 + 墙面喷涂覆盖
R = res{2};   % ±30°
r_cone = (WALL_X-WRIST_PLANE)*tand(FAN_ANGLE/2);
figure('Position',[80 80 1200 560],'Color','w');
subplot(1,2,1); hold on; box on; grid on; axis equal;
k=boundary(R.Y,R.Z,0.4); fill(R.Y(k),R.Z(k),[0.85 0.15 0.1],'FaceAlpha',0.2,'EdgeColor','none');
plot(R.Y,R.Z,'.','Color',[0.85 0.4 0.3],'MarkerSize',2);
k=boundary(R.Y,R.Z,0.4); plot(R.Y(k),R.Z(k),'Color',[0.85 0.15 0.1],'LineWidth',1.8);
plot([0 0],[0 2.2],'--','Color',[0.4 0.4 0.4]);
xlabel('墙面横向 y (m) (负=右)'); ylabel('墙面高度 z (m)');
title(sprintf('右臂腕部可达区(距墙40cm, 腰扭转±30°) 面积≈%.2f m^2',R.area));
text(mean([R.y0 R.y1]), max(R.Z)-0.03, sprintf('右偏中心 y=%.2f, 宽%.2fm×高%.2fm',mean([R.y0 R.y1]),range(R.Y),range(R.Z)), ...
    'HorizontalAlignment','center','FontSize',9);
% 喷涂覆盖
subplot(1,2,2); hold on; box on; grid on; axis equal;
th=linspace(0,2*pi,40); cx=r_cone*cos(th); cy=r_cone*sin(th);
step=max(1,round(numel(R.Y)/15000)); Yd=R.Y(1:step:end); Zd=R.Z(1:step:end);
for i=1:numel(Yd)
    patch(cx+Yd(i),cy+Zd(i),'c','EdgeColor','none','FaceAlpha',0.06);
end
% 喷涂覆盖面积(网格并集)
dy=0.02;dz=0.02; ymin=-1.0;ymax=1.0;zmin=0;zmax=2.2;
ny=round((ymax-ymin)/dy)+1; nz=round((zmax-zmin)/dz)+1;
rr=ceil(r_cone/dy); [ky,kz]=meshgrid(-rr:rr,-rr:rr); ky=ky(:);kz=kz(:);
keep=(ky.^2+kz.^2)<=(rr+0.5)^2; ky=ky(keep);kz=kz(keep);
spray=false(nz,ny);
for i=1:numel(Yd)
    c1=round((Yd(i)-ymin)/dy)+1; r1=round((Zd(i)-zmin)/dz)+1;
    for k=1:numel(ky)
        r2=r1+kz(k); c2=c1+ky(k);
        if r2>=1&&r2<=nz&&c2>=1&&c2<=ny, spray(r2,c2)=true; end
    end
end
area_spray=nnz(spray)*dy*dz;
plot([0 0],[0 2.2],'--','Color',[0.4 0.4 0.4]);
xlabel('墙面横向 y (m) (负=右)'); ylabel('墙面高度 z (m)');
title(sprintf('墙面喷涂覆盖(60°锥,距墙40cm,腰扭转±30°) 面积≈%.2f m^2',area_spray));
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','tiangong_workspace.png'));

%% 侧视(按URDF高度, 右臂前伸)
figure('Position',[420 300 640 460],'Color','w'); hold on; box on; grid on; axis equal;
xlim([-0.15 1.15]); ylim([-0.1 2.05]);
plot([-0.15 1.15],[0 0],'Color',[0.4 0.4 0.4],'LineWidth',2);
patch([WALL_X WALL_X WALL_X+0.03 WALL_X+0.03],[0 2.05 2.05 0],[0.85 0.9 0.95],'EdgeColor','none');
plot([WALL_X WALL_X],[0 2.05],'k-','LineWidth',2.5); text(WALL_X+0.02,2.08,'墙面 x=0.8m','FontSize',9);
plot([WRIST_PLANE WRIST_PLANE],[0 2.05],'--','Color',[0.6 0.6 0.6]);
text(WRIST_PLANE,2.08,'腕部/喷枪距墙40cm','FontSize',9,'HorizontalAlignment','center');
patch([-0.07 0.07 0.07 -0.07],[0 0 0.55 0.55],[0.75 0.78 0.82],'EdgeColor','none');
patch([-0.09 0.09 0.09 -0.09],[0.55 0.55 1.50 1.50],[0.55 0.62 0.7],'EdgeColor','none');
rectangle('Position',[-0.07 1.50 0.14 0.14],'Curvature',[1 1],'FaceColor',[0.85 0.8 0.7]);
plot([0 0],[1.50 1.63],'k-','LineWidth',2); text(0,1.66,'总高≈1.63m','FontSize',8,'HorizontalAlignment','center');
plot([-0.12 0.0],[0.96 0.96],'--','Color',[0.5 0.5 0.5]); text(-0.12,0.96,'骨盆0.96m','FontSize',8,'HorizontalAlignment','right');
plot([-0.12 0.0],[1.31 1.31],'--','Color',[0.5 0.5 0.5]); text(-0.12,1.31,'肩1.31m','FontSize',8,'HorizontalAlignment','right');
zw=0.95;
plot([0 WRIST_PLANE],[1.21 zw],'Color',[0.8 0.3 0.2],'LineWidth',3);
plot(WRIST_PLANE,zw,'o','MarkerSize',7,'MarkerFaceColor',[0.8 0.3 0.2]);
for zz=[0.6 0.95 1.3]
    rad=(WALL_X-WRIST_PLANE)*tand(FAN_ANGLE/2);
    plot([WRIST_PLANE WALL_X],[zz zz+rad],'Color',[0.9 0.5 0.2],'LineWidth',1.3);
    plot([WRIST_PLANE WALL_X],[zz zz-rad],'Color',[0.9 0.5 0.2],'LineWidth',1.3);
end
plot([0.05 WALL_X-0.05],[0.03 0.03],'k-','LineWidth',1.5);
text(mean([0 WALL_X]),0.07,'机器人中心距墙 80cm','HorizontalAlignment','center','FontSize',9);
xlabel('x (m)'); ylabel('z (m)');
title('侧视: 右臂喷枪垂直墙面, 距墙40cm, 60°锥形(URDF高度)');
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','tiangong_spray_side.png'));

%% 保存
save(fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','tiangong_ws_data.mat'), ...
    'res','r_cone','WALL_X','WRIST_PLANE','FAN_ANGLE');

%% ============ 右臂正向运动学 ============
function [wx,wy,wz]=TiangongArmFK_right(px,pz,q)
    T=eye(4); T(1:3,4)=[px;0;pz];
    T=T*urdf_joint([0 0 0],[0 0 0.0013415],[0 0 1],q(1));
    T=T*urdf_joint([0 0 0.083796],[0 0 0],[1 0 0],q(2));
    T=T*urdf_joint([0 0 0.078424],[0 0 0],[0 1 0],q(3));
    T=T*urdf_joint([0 -0.14552 0.18664],[-0.087266 0 0],[0 1 0],q(4));
    T=T*urdf_joint([0 -0.0678 0],[0 0 0],[1 0 0],q(5));
    T=T*urdf_joint([0 0 -0.1004],[0 0 0],[0 0 1],q(6));
    T=T*urdf_joint([0.02 0 -0.1596],[0 0 0],[0 1 0],q(7));
    R=T(1:3,1:3); p=T(1:3,4); wp=p+R*[0;0;-0.222];
    wx=wp(1);wy=wp(2);wz=wp(3);
end
function T=urdf_joint(o,rpy,axis,q)
    Rrpy=rotz_own(rpy(3))*roty_own(rpy(2))*rotx_own(rpy(1));
    T=eye(4); T(1:3,1:3)=Rrpy; T(1:3,4)=o(:);
    aa=axis/norm(axis); c=cos(q); s=sin(q); C=1-c;
    Ra=[aa(1)^2*C+c, aa(1)*aa(2)*C-aa(3)*s, aa(1)*aa(3)*C+aa(2)*s;
        aa(2)*aa(1)*C+aa(3)*s, aa(2)^2*C+c, aa(2)*aa(3)*C-aa(1)*s;
        aa(3)*aa(1)*C-aa(2)*s, aa(3)*aa(2)*C+aa(1)*s, aa(3)^2*C+c];
    T=T*[Ra zeros(3,1);0 0 0 1];
end
function R=rotx_own(t),c=cos(t);s=sin(t);R=[1 0 0;0 c -s;0 s c];end
function R=roty_own(t),c=cos(t);s=sin(t);R=[c 0 s;0 1 0;-s 0 c];end
function R=rotz_own(t),c=cos(t);s=sin(t);R=[c -s 0;s c 0;0 0 1];end