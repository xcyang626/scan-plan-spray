%% 舱室绝缘喷涂 - 工位区域划分(牛耕分解法BCD) & 工作点规划
% 墙面: 宽10m×高2m, 机器人中心距墙0.8m, 腕部/喷枪距墙0.4m
% 障碍物: 栅栏1@4m(高30cm宽10cm), 栅栏2@7m, 控制台@9.2-10m(宽80cm高1m, 加强筋@1.4m)
% 区域划分: 牛耕法(BCD), 栅栏上方区域归入左侧大区域
% 工作点: 按机器人单工位覆盖(约1.1m宽×1.6m高)划分, 角落40cm处单独设点, 加强筋单独设点
clear; clc; close all;

%% ===== 参数 =====
WALL_W = 10; WALL_H = 2;        % 墙面 10m×2m
ROBOT_W = 0.80;                  % 机器人中心距墙
GUN_DIST = 0.40;                % 腕部/喷枪距墙

% 栅栏1 @4m, 栅栏2 @7m (高30cm, 宽10cm, 地面起点)
fence1_y = 4.0; fence_w = 0.10; fence_h = 0.30;
fence2_y = 7.0;

% 控制台 @9.2-10m, 宽80cm, 高1m; 加强筋@1.4m (厚10cm宽30cm)
console_y0 = 9.2; console_w = 0.80; console_h = 1.0;
stiff_h = 1.4; stiff_w = 0.30; stiff_t = 0.10;

% 机器人单工位覆盖(Part3结果: 腰扭转±30°, 右臂可达区)
ws_w = 1.1; ws_h = 1.6;          % 单工位覆盖宽×高

%% ===== Part4: 牛耕法(BCD)区域划分 =====
% 障碍物投影到墙面, 切割线由障碍物边界决定
% 切割线 y 坐标: 栅栏左右边 + 控制台左边界
cut_lines = [fence1_y, fence1_y+fence_w, fence2_y, fence2_y+fence_w, console_y0];
cut_lines = unique(cut_lines);

% BCD 区域划分
% 区域1: y=[0, 4.1] (含栅栏1上方, 归入左侧)
% 区域2: y=[4.1, 7.1] (含栅栏2上方, 归入左侧)
% 区域3: y=[7.1, 9.2]
% 区域4: y=[9.2, 10.0] (控制台, z=[1.0, 2.0])
regions = struct('name',{},'y0',{},'y1',{},'z0',{},'z1',{},'color',{},'width',{},'height',{});
regions(1)=struct('name','区域1','y0',0,'y1',4.1,'z0',0,'z1',2.0,'color',[0.85 0.92 0.98],'width',4.1,'height',2.0);
regions(2)=struct('name','区域2','y0',4.1,'y1',7.1,'z0',0,'z1',2.0,'color',[0.92 0.98 0.85],'width',3.0,'height',2.0);
regions(3)=struct('name','区域3','y0',7.1,'y1',9.2,'z0',0,'z1',2.0,'color',[0.98 0.92 0.85],'width',2.1,'height',2.0);
regions(4)=struct('name','区域4','y0',9.2,'y1',10.0,'z0',1.0,'z1',2.0,'color',[0.95 0.88 0.92],'width',0.8,'height',1.0);

%% 图1: 墙面区域划分(BCD)
figure('Position',[60 200 1100 350],'Color','w');
hold on; box on; grid on; axis equal;
xlim([-0.3 10.5]); ylim([-0.3 2.6]);

% 墙面底框
patch([0 WALL_W WALL_W 0],[0 0 WALL_H WALL_H],[0.95 0.95 0.95],'EdgeColor',[0.3 0.3 0.3]);

% 绘制各区域(半透明)
for i=1:numel(regions)
    r=regions(i);
    fill([r.y0 r.y1 r.y1 r.y0],[r.z0 r.z0 r.z1 r.z1],r.color,'FaceAlpha',0.5,'EdgeColor',[0.2 0.2 0.2],'LineWidth',1.5);
    text((r.y0+r.y1)/2, (r.z0+r.z1)/2, r.name, 'FontSize',12,'FontWeight','bold','HorizontalAlignment','center');
end

% 栅栏1 (y=4.0-4.1, z=0-0.3)
patch([fence1_y fence1_y+fence_w fence1_y+fence_w fence1_y],[0 0 fence_h fence_h],[0.85 0.2 0.2],'EdgeColor','k','LineWidth',1.2);
text(fence1_y+fence_w/2, -0.15, sprintf('栅栏1\n%.1fm',fence1_y),'FontSize',9,'HorizontalAlignment','center','Color',[0.7 0.1 0.1]);

% 栅栏2 (y=7.0-7.1, z=0-0.3)
patch([fence2_y fence2_y+fence_w fence2_y+fence_w fence2_y],[0 0 fence_h fence_h],[0.85 0.2 0.2],'EdgeColor','k','LineWidth',1.2);
text(fence2_y+fence_w/2, -0.15, sprintf('栅栏2\n%.1fm',fence2_y),'FontSize',9,'HorizontalAlignment','center','Color',[0.7 0.1 0.1]);

% 控制台 (y=9.2-10.0, z=0-1.0) - 灰色
patch([console_y0 WALL_W WALL_W console_y0],[0 0 console_h console_h],[0.55 0.55 0.55],'EdgeColor','k','LineWidth',1.2);
text(console_y0+console_w/2, console_h/2, '控制台', 'FontSize',10,'FontWeight','bold','HorizontalAlignment','center','Color','w');

% 加强筋 (y=9.2-9.5, z=1.4, 厚10cm宽30cm)
stiff_y0 = console_y0; stiff_y1 = console_y0 + stiff_w;
patch([stiff_y0 stiff_y1 stiff_y1 stiff_y0],[stiff_h stiff_h stiff_h+stiff_t stiff_h+stiff_t],[0.9 0.6 0.2],'EdgeColor','k','LineWidth',1.2);
text(stiff_y0+stiff_w/2, stiff_h+stiff_t+0.1, '加强筋', 'FontSize',8,'HorizontalAlignment','center','Color',[0.6 0.3 0.05]);

% BCD切割线
for cy=cut_lines
    plot([cy cy],[0 WALL_H],'--','Color',[0.3 0.3 0.3],'LineWidth',1.2);
end

% 栅栏上方区域标注(归入左侧)
% 栅栏1上方
plot([fence1_y fence1_y+fence_w],[fence_h fence_h],'r-','LineWidth',1.5);
text(fence1_y+fence_w/2, fence_h+0.1, '\leftarrow归入区域1', 'FontSize',8,'Color','r','HorizontalAlignment','center');
% 栅栏2上方
plot([fence2_y fence2_y+fence_w],[fence_h fence_h],'r-','LineWidth',1.5);
text(fence2_y+fence_w/2, fence_h+0.1, '\leftarrow归入区域2', 'FontSize',8,'Color','r','HorizontalAlignment','center');

% 尺寸标注
for i=1:numel(regions)
    r=regions(i);
    plot([r.y0 r.y1],[r.z1+0.05 r.z1+0.05],'k-','LineWidth',1);
    plot([r.y0 r.y0],[r.z1 r.z1+0.05],'k-'); plot([r.y1 r.y1],[r.z1 r.z1+0.05],'k-');
    text((r.y0+r.y1)/2, r.z1+0.15, sprintf('%.1fm',r.width), 'FontSize',9,'HorizontalAlignment','center');
end
% 控制台宽度
plot([console_y0 WALL_W],[console_h+0.05 console_h+0.05],'k-','LineWidth',1);
text(console_y0+console_w/2, console_h+0.2, sprintf('%.1fm',console_w), 'FontSize',9,'HorizontalAlignment','center');

% 图例/标注
text(0.3, 2.4, '墙面: 10m×2m', 'FontSize',11,'FontWeight','bold');
text(0.3, 2.2, '▬ 栅栏(30cm高×10cm宽)', 'FontSize',9,'Color',[0.7 0.1 0.1]);
text(4.5, 2.2, '▬ 控制台(80cm×1m) + 加强筋', 'FontSize',9,'Color',[0.55 0.55 0.55]);
text(0.3, 2.0, '▬ BCD切割线', 'FontSize',9,'Color',[0.3 0.3 0.3]);

xlabel('墙面横向 y (m)'); ylabel('墙面高度 z (m)');
title('工位区域划分(牛耕分解法BCD) - 栅栏上方归入左侧大区域');
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','bcd_regions.png'));

%% ===== Part5: 工作点划分 =====
% 每个区域按机器人单工位覆盖(ws_w×ws_h)划分
% 角落: 距墙端40cm处单独设点, 朝向墙角
% 加强筋: 单独设点, 朝向加强筋位置

corner_offset = 0.40;  % 角落站位距墙边40cm

stations = struct('x',{},'y',{},'z',{},'name',{},'type',{},'cover_y0',{},'cover_y1',{},'cover_z0',{},'cover_z1',{});

% 区域1: y=[0, 4.1], z=[0, 2.0], 宽4.1m -> 4 sub-regions
r = regions(1);
n_cols = ceil(r.width / ws_w);
sub_w = r.width / n_cols;
for c = 1:n_cols
    cy0 = r.y0 + (c-1)*sub_w; cy1 = cy0 + sub_w;
    st_y = (cy0+cy1)/2;
    st_x = ROBOT_W;  % 机器人站位于墙前方80cm
    st_z = 0;
    n = length(stations)+1;
    stations(n) = struct('x',st_x,'y',st_y,'z',st_z,'name',sprintf('R1-%d',c),...
        'type','normal','cover_y0',cy0,'cover_y1',cy1,'cover_z0',r.z0,'cover_z1',r.z1);
end

% 区域2: y=[4.1, 7.1], 宽3.0m -> 3 sub-regions
r = regions(2);
n_cols = ceil(r.width / ws_w);
sub_w = r.width / n_cols;
for c = 1:n_cols
    cy0 = r.y0 + (c-1)*sub_w; cy1 = cy0 + sub_w;
    st_y = (cy0+cy1)/2;
    st_x = ROBOT_W;
    st_z = 0;
    n = length(stations)+1;
    stations(n) = struct('x',st_x,'y',st_y,'z',st_z,'name',sprintf('R2-%d',c),...
        'type','normal','cover_y0',cy0,'cover_y1',cy1,'cover_z0',r.z0,'cover_z1',r.z1);
end

% 区域3: y=[7.1, 9.2], 宽2.1m -> 2 sub-regions
r = regions(3);
n_cols = ceil(r.width / ws_w);
sub_w = r.width / n_cols;
for c = 1:n_cols
    cy0 = r.y0 + (c-1)*sub_w; cy1 = cy0 + sub_w;
    st_y = (cy0+cy1)/2;
    st_x = ROBOT_W;
    st_z = 0;
    n = length(stations)+1;
    stations(n) = struct('x',st_x,'y',st_y,'z',st_z,'name',sprintf('R3-%d',c),...
        'type','normal','cover_y0',cy0,'cover_y1',cy1,'cover_z0',r.z0,'cover_z1',r.z1);
end

% 区域4: 控制台 y=[9.2, 10.0], z=[1.0, 2.0], 宽0.8m -> 1 sub-region, 但需特殊处理
r = regions(4);
for c = 1:1
    cy0 = r.y0; cy1 = r.y1;
    st_y = (cy0+cy1)/2;
    st_x = ROBOT_W;
    st_z = 0;
    n = length(stations)+1;
    stations(n) = struct('x',st_x,'y',st_y,'z',st_z,'name','R4-1',...
        'type','console','cover_y0',cy0,'cover_y1',cy1,'cover_z0',r.z0,'cover_z1',r.z1);
end

% 角落站位: 左端40cm (y=0.4), 朝向左侧墙角
n = length(stations)+1;
stations(n) = struct('x',ROBOT_W,'y',corner_offset,'z',0,'name','角L',...
    'type','corner','cover_y0',0,'cover_y1',corner_offset+0.3,'cover_z0',0,'cover_z1',WALL_H);
% 右端40cm (y=9.6), 朝向右侧墙角
n = length(stations)+1;
stations(n) = struct('x',ROBOT_W,'y',WALL_W-corner_offset,'z',0,'name','角R',...
    'type','corner','cover_y0',WALL_W-corner_offset-0.3,'cover_y1',WALL_W,'cover_z0',0,'cover_z1',WALL_H);

% 加强筋单独站位: 在加强筋位置(y=9.2-9.5, z=1.4)
n = length(stations)+1;
stations(n) = struct('x',ROBOT_W,'y',stiff_y0+stiff_w/2,'z',0,'name','加强筋',...
    'type','stiffener','cover_y0',stiff_y0,'cover_y1',stiff_y1,'cover_z0',stiff_h,'cover_z1',stiff_h+stiff_t+0.1);

fprintf('总站位点数: %d\n', length(stations));
fprintf('  普通: %d, 角落: %d, 加强筋: %d, 控制台: %d\n', ...
    sum(strcmp({stations.type},'normal')), sum(strcmp({stations.type},'corner')), ...
    sum(strcmp({stations.type},'stiffener')), sum(strcmp({stations.type},'console')));

%% 图2: 工作点划分(墙面俯视+侧视)
figure('Position',[60 60 1150 500],'Color','w');
subplot(1,2,1); hold on; box on; grid on; axis equal;
xlim([-0.5 10.8]); ylim([-0.5 3.5]);
% 墙面线
plot([0 WALL_W],[0 0],'k-','LineWidth',3);
plot([0 WALL_W],[WALL_H WALL_H],'k-','LineWidth',1.5);
plot([0 0],[0 WALL_H],'k-'); plot([WALL_W WALL_W],[0 WALL_H],'k-');
% 各区域
for i=1:numel(regions)
    r=regions(i);
    fill([r.y0 r.y1 r.y1 r.y0],[r.z0 r.z0 r.z1 r.z1],r.color,'FaceAlpha',0.3);
end
% 障碍物
patch([fence1_y fence1_y+fence_w fence1_y+fence_w fence1_y],[0 0 fence_h fence_h],[0.85 0.2 0.2]);
patch([fence2_y fence2_y+fence_w fence2_y+fence_w fence2_y],[0 0 fence_h fence_h],[0.85 0.2 0.2]);
patch([console_y0 WALL_W WALL_W console_y0],[0 0 console_h console_h],[0.55 0.55 0.55]);
patch([stiff_y0 stiff_y1 stiff_y1 stiff_y0],[stiff_h stiff_h stiff_h+stiff_t stiff_h+stiff_t],[0.9 0.6 0.2]);
% 站位点投影
for i=1:numel(stations)
    s=stations(i);
    switch s.type
        case 'normal',   m='s'; mc=[0.2 0.5 0.8]; ms=10;
        case 'corner',   m='d'; mc=[0.85 0.2 0.1]; ms=12;
        case 'stiffener',m='^'; mc=[0.9 0.5 0.1]; ms=12;
        case 'console',  m='o'; mc=[0.3 0.3 0.3]; ms=10;
    end
    % 站位点投影到墙面(y,z)
    plot(s.y, (s.cover_z0+s.cover_z1)/2, m, 'MarkerSize',ms, 'MarkerFaceColor',mc, 'MarkerEdgeColor','k');
    text(s.y, (s.cover_z0+s.cover_z1)/2+0.1, s.name, 'FontSize',7,'HorizontalAlignment','center');
end
xlabel('墙面横向 y (m)'); ylabel('墙面高度 z (m)');
title('工作点划分(墙面投影)');
legend({'普通','角落','加强筋','控制台'},'Location','northeast');

%% 图3: 三维站位点示意图
figure('Position',[100 60 1000 600],'Color','w');
hold on; box on; grid on; view(35,25); axis equal;
xlim([-0.5 2.5]); ylim([-0.5 10.5]); zlim([-0.2 2.5]);
xlabel('x (距墙)'); ylabel('y (墙面横向)'); zlabel('z (高度)');

% 墙面(半透明)
patch([ROBOT_W-GUN_DIST ROBOT_W-GUN_DIST ROBOT_W-GUN_DIST ROBOT_W-GUN_DIST], ...
    [0 WALL_W WALL_W 0], [0 0 WALL_H WALL_H], [0.7 0.8 0.95], 'FaceAlpha',0.4,'EdgeColor','k');

% 地面
patch([0 2 2 0],[0 0 WALL_W WALL_W],[0 0 0 0],[0.85 0.85 0.85],'FaceAlpha',0.5);

% 障碍物(3D)
% 栅栏1
bar3d(fence1_y, fence1_y+fence_w, 0, fence_h, GUN_DIST, [0.85 0.2 0.2]);
% 栅栏2
bar3d(fence2_y, fence2_y+fence_w, 0, fence_h, GUN_DIST, [0.85 0.2 0.2]);
% 控制台
bar3d(console_y0, WALL_W, 0, console_h, GUN_DIST, [0.55 0.55 0.55]);
% 加强筋
bar3d(stiff_y0, stiff_y1, stiff_h, stiff_h+stiff_t, GUN_DIST, [0.9 0.6 0.2]);

% 站位点(3D)
for i=1:numel(stations)
    s=stations(i);
    switch s.type
        case 'normal',   mc=[0.2 0.5 0.8];
        case 'corner',   mc=[0.85 0.2 0.1];
        case 'stiffener',mc=[0.9 0.5 0.1];
        case 'console',  mc=[0.3 0.3 0.3];
    end
    % 机器人站位点(地面上, 距墙ROBOT_W)
    plot3(s.x, s.y, 0.05, 'o', 'MarkerSize',10, 'MarkerFaceColor',mc, 'MarkerEdgeColor','k');
    % 机器人躯干示意(竖线)
    plot3([s.x s.x],[s.y s.y],[0.05 1.63],'Color',mc,'LineWidth',2);
    % 从站位到腕部平面的连线
    plot3([s.x GUN_DIST],[s.y s.y],[1.0 1.0],'--','Color',mc,'LineWidth',0.8);
    text(s.x+0.05, s.y, 0.15, s.name, 'FontSize',7,'Color',mc);
end

% 角落朝向标注
% 左角: 站位在y=0.4, 朝向墙角(0,0)
sL = stations(end-1);
quiver3(sL.x, sL.y, 0.5, -0.3, -0.3, 0, 0.3, 'Color',[0.85 0.2 0.1],'LineWidth',2,'MaxHeadSize',0.5);
% 右角: 站位在y=9.6, 朝向墙角(WALL_W,0)
sR = stations(end);
quiver3(sR.x, sR.y, 0.5, -0.3, 0.3, 0, 0.3, 'Color',[0.85 0.2 0.1],'LineWidth',2,'MaxHeadSize',0.5);

title(sprintf('三维站位点示意图(共%d个站位点)', length(stations)));
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','station_3d.png'));

%% 图4: 工作点划分(墙面俯视图, 含覆盖区域)
figure('Position',[60 60 1000 480],'Color','w');
hold on; box on; grid on; axis equal;
xlim([-0.8 2.0]); ylim([-0.5 10.5]);
% 墙面线
plot([0 0],[0 WALL_W],'k-','LineWidth',2);
% 地面
patch([0 2 2 0],[0 0 WALL_W WALL_W],[0.9 0.9 0.9],'FaceAlpha',0.3);
% 障碍物 (俯视: x=0到障碍物宽度, y=障碍物沿墙位置)
patch([0 fence_w fence_w 0],[fence1_y fence1_y fence1_y+fence_w fence1_y+fence_w],[0.85 0.2 0.2],'FaceAlpha',0.7);
patch([0 fence_w fence_w 0],[fence2_y fence2_y fence2_y+fence_w fence2_y+fence_w],[0.85 0.2 0.2],'FaceAlpha',0.7);
patch([0 console_w console_w 0],[console_y0 console_y0 WALL_W WALL_W],[0.55 0.55 0.55],'FaceAlpha',0.7);
% 站位点+覆盖区域
for i=1:numel(stations)
    s=stations(i);
    switch s.type
        case 'normal',   mc=[0.2 0.5 0.8]; fa=0.15;
        case 'corner',   mc=[0.85 0.2 0.1]; fa=0.2;
        case 'stiffener',mc=[0.9 0.5 0.1]; fa=0.25;
        case 'console',  mc=[0.3 0.3 0.3]; fa=0.2;
    end
    % 覆盖区域(墙面上的投影)
    fill([0 0 GUN_DIST GUN_DIST],[s.cover_y0 s.cover_y1 s.cover_y1 s.cover_y0],mc,'FaceAlpha',fa,'EdgeColor',mc,'LineWidth',1);
    % 站位点
    plot(s.x, s.y, 'o', 'MarkerSize',8, 'MarkerFaceColor',mc, 'MarkerEdgeColor','k');
    text(s.x+0.05, s.y, s.name, 'FontSize',7,'Color',mc);
end
% 机器人中心距墙线
plot([ROBOT_W ROBOT_W],[0 WALL_W],'--','Color',[0.5 0.5 0.5],'LineWidth',1);
text(ROBOT_W, -0.35, '机器人站位线', 'FontSize',8,'HorizontalAlignment','center');
xlabel('x (距墙)'); ylabel('y (墙面横向)');
title('俯视: 站位点分布与墙面覆盖区域');
saveas(gcf, fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','station_topview.png'));

%% 保存数据
save(fullfile('c:\Users\Yangon\.trae-cn\work\6a7d8c106f3fddaa48e63a71','station_data.mat'), ...
    'regions','stations','WALL_W','WALL_H','ROBOT_W','GUN_DIST','ws_w','ws_h');

fprintf('\n===== 区域划分与工作点规划完成 =====\n');
fprintf('区域数: %d, 总站位点: %d\n', numel(regions), length(stations));
for i=1:numel(regions)
    r=regions(i);
    fprintf('  %s: y=[%.1f,%.1f] z=[%.1f,%.1f] 宽%.1fm×高%.1fm\n', r.name, r.y0, r.y1, r.z0, r.z1, r.width, r.height);
end

%% ===== 辅助函数 =====
function bar3d(y0, y1, z0, z1, thick, col)
    verts = [0 y0 z0; 0 y1 z0; 0 y1 z1; 0 y0 z1;
             thick y0 z0; thick y1 z0; thick y1 z1; thick y0 z1];
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    patch('Vertices',verts,'Faces',faces,'FaceColor',col,'EdgeColor','k','FaceAlpha',0.8);
end