import svgwrite
import cairosvg
import base64

views = [
    "surface.png", "sticks.png", "mesh.png", "cartoon.png",
    "surface_0.png", "surface_90.png", "surface_180.png", "surface_270.png"
]

output_svg = "combined_figure.svg"
output_png = "combined_figure.png"

canvas_width, canvas_height = 1200, 700
top_y, bottom_y = 20, 330
img_width, img_height = 250, 200
img_width_bottom, img_height_bottom = 250, 250
x_positions = [60, 340, 620, 900]

dwg = svgwrite.Drawing(output_svg, size=(canvas_width, canvas_height))
# 背景填充白色
dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

dwg.defs.add(dwg.style("""
.label { font-family: Arial; font-size:20px; fill:black; }
.sub { font-family: Arial; font-size:18px; font-weight:bold; fill:black; }
.angle { font-family: Arial; font-size:12px; fill:#4aa3d8; }
.dashedbox { fill:none; stroke:#2b2b2b; stroke-width:2; stroke-dasharray:6 6;
             rx:18; ry:18; stroke-linecap:round; stroke-linejoin:round; }
"""))

def embed_png(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# 绘制大框
dwg.add(dwg.rect(insert=(30, top_y), size=(1140, 280), class_="dashedbox"))
dwg.add(dwg.text("a", insert=(40, 40), class_="sub"))
dwg.add(dwg.rect(insert=(30, bottom_y), size=(1140, 320), class_="dashedbox"))
dwg.add(dwg.text("b", insert=(40, bottom_y+20), class_="sub"))

# 顶部四张图
labels = ["Surface", "Stick", "Mesh", "Cartoon"]
for i in range(4):
    dwg.add(dwg.image(embed_png(views[i]), insert=(x_positions[i], 60), size=(img_width, img_height)))
    dwg.add(dwg.text(labels[i], insert=(x_positions[i] + img_width/2, 55),
                     class_="label", text_anchor="middle"))

# 底部四张图
angles = ["0°", "90°", "180°", "270°"]
for i in range(4):
    dwg.add(dwg.image(embed_png(views[i+4]), insert=(x_positions[i], 380), size=(img_width_bottom, img_height_bottom)))
    dwg.add(dwg.text(angles[i], insert=(x_positions[i] + img_width_bottom/2, 370),
                     class_="angle", text_anchor="middle"))

dwg.save()
cairosvg.svg2png(url=output_svg, write_to=output_png)
print(f"✅ 已生成白色背景 + 圆头的 {output_svg} 和 {output_png}")
