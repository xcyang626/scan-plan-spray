# 喷涂轨迹工作站 · 界面设计规范 (参考 Linear / Vercel / Stripe Dashboard / Grafana)

所有值可直接落 CSS。核心原则: **少即多**——3 层表面灰阶建立纵深, 一个品牌色统领,
动效只用于"状态变化"与"入场", 150–250ms, 统一缓动。

## 1. 色彩 Token

| Token | 值 | 用途 |
|---|---|---|
| --bg0 | #0b0f14 | 页面底色 |
| --bg1 | #11161d | 卡片表面 |
| --bg2 | #171e27 | 卡片内嵌表面(日志/画布/输入框) |
| --line | #1f2937 | 边框(默认) |
| --line2 | #2c3a4d | 边框(hover/强调) |
| --text | #e6edf3 | 主文字 |
| --text2 | #9fb0c3 | 次文字 |
| --text3 | #66788c | 弱文字/图注 |
| --brand | #4f8ef7 | 主色(主按钮/链接/进度) |
| --brand2 | #3b6fd4 | 主色 hover |
| --ok | #3fce7a | 成功/通过 |
| --warn | #f5b043 | 待检查/进行中 |
| --err | #f5616c | 失败 |

## 2. 字体

```css
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
     "Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
     font-size:14px; line-height:1.55; color:var(--text);
     -webkit-font-smoothing:antialiased;}
h1{font-size:17px;font-weight:650;letter-spacing:.2px}
.card-title{font-size:13.5px;font-weight:600;color:var(--text)}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"} /* 所有数字/指标加此类 */
.aux{font-size:12px;color:var(--text2)}
.code,pre{font-family:"JetBrains Mono","SF Mono",Consolas,"Courier New",monospace;
          font-size:11.5px;line-height:1.5}
```

## 3. 间距 / 圆角 / 阴影

- 卡片 padding 16px; 栅格 gap 14px; 卡片圆角 **12px**; 按钮/输入 8px; 徽章 999px
- 卡片阴影: `box-shadow:0 1px 2px rgba(0,0,0,.35),0 0 0 1px var(--line)`
- hover 提亮: 边框→var(--line2) + `box-shadow:0 6px 18px rgba(0,0,0,.35)`

## 4. 按钮三态

```css
button{transition:transform .15s cubic-bezier(.2,.8,.2,1),box-shadow .15s,filter .15s}
.btn-primary{background:linear-gradient(180deg,#5b97ff,#3b6fd4);border:0;color:#fff;
  box-shadow:0 1px 0 rgba(255,255,255,.15) inset,0 4px 14px rgba(59,111,212,.35)}
.btn-primary:hover{filter:brightness(1.08);transform:translateY(-1px)}
.btn-primary:active{transform:translateY(0);filter:brightness(.95)}
.btn-ghost{background:var(--bg2);border:1px solid var(--line);color:var(--text)}
.btn-ghost:hover{border-color:var(--line2);transform:translateY(-1px)}
button:disabled{background:#1a2230;color:#5b6b7e;box-shadow:none;transform:none;cursor:not-allowed}
```

## 5. 动画 (全部 150–250ms, 缓动 var(--ease))

```css
:root{--ease:cubic-bezier(.2,.8,.2,1)}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:.55}50%{opacity:1}}
@keyframes shine{from{background-position:-200% 0}to{background-position:200% 0}}
@keyframes lbIn{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
/* 卡片入场 stagger: */
.card{animation:rise .3s var(--ease) both}
.card:nth-child(2){animation-delay:.06s}
/* 检查区出现时逐条上浮: */
.sumitem,.figcell{animation:rise .35s var(--ease) both}
.figcell:nth-child(2){animation-delay:.05s} /* …依次 +0.05s */
/* 生成中: 状态文字旁呼吸圆点 / 按钮扫光 */
.busy-dot{width:8px;height:8px;border-radius:99px;background:var(--warn);animation:pulse 1.2s infinite}
.btn-busy{background:linear-gradient(100deg,#3b6fd4 40%,#5b97ff 50%,#3b6fd4 60%);
          background-size:200% 100%;animation:shine 1.4s linear infinite}
/* 灯箱 */
#lightbox{animation:lbIn .2s var(--ease)}
/* 图片载入淡入 */
.figcell img{opacity:0;transition:opacity .4s ease}
.figcell img.ready{opacity:1}  /* onload 加类 */
```

## 6. 状态徽章 (替代 emoji ✅)

```css
.bdg{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;
     border-radius:99px;vertical-align:-4px;margin-right:2px}
.bdg::after{content:"";width:9px;height:5px;border-left:2px solid #08120b;border-bottom:2px solid #08120b;
     transform:rotate(-45deg) translate(1px,-1px)}
.bdg-ok{background:var(--ok)}            /* 绿底对勾 */
.bdg-bad{background:var(--err)}          /* 红底, ::after 换成 ✕: 两条 2px 线 rotate ±45 */
.bdg-bad::after{width:9px;height:9px;border:0;background:
     linear-gradient(45deg,transparent 42%,#2a0a0d 42%,#2a0a0d 58%,transparent 58%),
     linear-gradient(-45deg,transparent 42%,#2a0a0d 42%,#2a0a0d 58%,transparent 58%)}
/* 流程步骤条(顶栏下): ①输入 → ②轨迹检查 → ③动作 → ④执行包 */
.step{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text3)}
.step .dot{width:10px;height:10px;border-radius:99px;background:#2a3646}
.step.active{color:var(--warn)}  .step.active .dot{background:var(--warn);animation:pulse 1.2s infinite}
.step.done{color:var(--ok)}      .step.done .dot{background:var(--ok)}
```

## 7. 图片网格

```css
.figcell{border-radius:10px;overflow:hidden;background:#fff;border:1px solid var(--line);
         transition:transform .18s var(--ease),box-shadow .18s,border-color .18s;position:relative}
.figcell:hover{transform:translateY(-3px);border-color:var(--line2);
               box-shadow:0 10px 24px rgba(0,0,0,.4)}
.figcell .zoom{position:absolute;right:8px;bottom:8px;width:26px;height:26px;border-radius:8px;
         background:rgba(11,15,20,.72);display:flex;align-items:center;justify-content:center;
         opacity:0;transition:opacity .18s}      /* 内放 SVG 放大镜(白色描边) */
.figcell:hover .zoom{opacity:1}
.figcap{font-size:11.5px;color:var(--text3);padding:6px 8px;background:var(--bg1)}
```

## 8. 其他

- 顶栏: 高 52px, 底部 1px var(--line), 品牌名后加小圆点 var(--brand)
- 滚动条: `::-webkit-scrollbar{width:10px} thumb{background:#263242;border-radius:99px;border:3px solid var(--bg1)}`
- 日志 pre: 背景 var(--bg2), 边框 var(--line), 自动滚底; 最新行不需要高亮
- 结论条(sumitem): 背景 var(--bg2), 数字用 .num, 通过项徽章 .bdg-ok
