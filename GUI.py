import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QSlider, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt  # 需要导入 Qt
from PIL import Image
from pytorch_lightning import seed_everything
import einops
from ldm_hacked import *
import random


# generate_image(prompt, a_prompt, n_prompt) 函数来生成图像
def generate_image(prompt_user, a_prompt_user, n_prompt_user,ddim_steps_user,name_img):
    # 用Stable Diffusion的代码来生成图像
    # ----------------------- #
    #   使用的参数
    # ----------------------- #
    # config的地址
    config_path = "model_data/sd_v15.yaml"
    # 模型的地址
    model_path = "model_data/v1-5-pruned-emaonly.safetensors"
    # fp16，可以加速与节省显存
    sd_fp16 = True
    vae_fp16 = True
    #   保存路径
    # ----------------------- #
    save_path = "imgs/outputs_images"

    # ----------------------- #
    #   生成图片的参数
    # ----------------------- #
    # 生成的图像大小为input_shape，对于img2img会进行Centter Crop
    input_shape = [512, 512]
    # 一次生成几张图像
    num_samples = 1
    # 采样的步数
    ddim_steps = ddim_steps_user
    # 采样的种子，为-1的话则随机。
    seed = 12345
    # eta
    eta = 0
    # denoise强度，for img2img
    denoise_strength = 1.00
    # ----------------------- #
    #   提示词相关参数
    # ----------------------- #
    # 提示词
    prompt = prompt_user
    # 正面提示词
    a_prompt = a_prompt_user
    # 负面提示词
    n_prompt = n_prompt_user
    print("prompt:", prompt)
    print("a_prompt:", a_prompt)
    print("n_prompt:", n_prompt)
    # 正负扩大倍数
    scale = 9
    # img2img使用，如果不想img2img这设置为None。
    image_path = None
    # inpaint使用，如果不想inpaint这设置为None；inpaint使用需要结合img2img。
    # 注意mask图和原图需要一样大
    mask_path = None
    # ----------------------- #
    #   创建模型
    # ----------------------- #
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    if sd_fp16:
        model = model.half()

    with torch.no_grad():
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # ----------------------- #
        #   获得编码后的prompt
        # ----------------------- #
        cond = {"c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        H, W = input_shape
        shape = (4, H // 8, W // 8)
        # ----------------------- #
        #   进行采样
        # ----------------------- #
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        # ----------------------- #
        #   进行解码
        # ----------------------- #
        x_samples = model.decode_first_stage(samples.half() if vae_fp16 else samples.float())
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for index, image in enumerate(x_samples):
        cv2.imwrite(os.path.join(save_path, name_img + ".jpg"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return x_samples


class StableDiffusionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置主窗口
        self.setWindowTitle('Stable Diffusion GUI -- Tao Yueming')
        self.setGeometry(100, 100, 1500, 1100)

        # 创建主布局
        main_layout = QHBoxLayout()

        # 左侧的文本输入区域布局
        input_layout = QVBoxLayout()

        # 滑动条标题
        slider_label = QLabel("采样步数 (DDIM Steps):", self)
        input_layout.addWidget(slider_label)
        # 创建一个 QLabel 用于显示滑动条的当前值
        self.slider_value_label = QLabel(f"当前步数: {40}", self)  # 初始化时设置为滑动条的默认值 160
        input_layout.addWidget(self.slider_value_label)

        # 添加滑动条来控制 `ddim_steps`
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(10)
        self.slider.setMaximum(200)
        self.slider.setValue(40)  # 设置默认值为40
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_slider_value)  # 连接滑动条值改变信号
        input_layout.addWidget(self.slider)

        # 提示词标题
        prompt_label = QLabel("提示词 (Prompt):", self)
        input_layout.addWidget(prompt_label)
        # 提示词输入框
        self.prompt_edit = QTextEdit(self)
        self.prompt_edit.setPlaceholderText("a cute cat, with yellow leaf, trees")
        input_layout.addWidget(self.prompt_edit)

        # 正面提示词标题
        a_prompt_label = QLabel("正面提示词 (Positive Prompt):", self)
        input_layout.addWidget(a_prompt_label)
        # 正面提示词输入框
        self.a_prompt_edit = QTextEdit(self)
        self.a_prompt_edit.setPlaceholderText("best quality, extremely detailed")
        input_layout.addWidget(self.a_prompt_edit)

        # 负面提示词标题
        n_prompt_label = QLabel("负面提示词 (Negative Prompt):", self)
        input_layout.addWidget(n_prompt_label)
        # 负面提示词输入框
        self.n_prompt_edit = QTextEdit(self)
        self.n_prompt_edit.setPlaceholderText("longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
        input_layout.addWidget(self.n_prompt_edit)

        # 图像保存名称
        name_label = QLabel("图像名称 (Name):", self)
        input_layout.addWidget(name_label)
        self.name_label = QTextEdit(self)
        self.name_label.setPlaceholderText("name")
        input_layout.addWidget(self.name_label)

        main_layout.addLayout(input_layout)

        # 右侧的布局
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        # 创建一个水平布局来放置按钮
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)  # Center the button within this layout

        # 添加按钮
        self.generate_button = QPushButton('生成图像', self)
        self.generate_button.setFixedSize(200, 60)  # 设置按钮的固定大小 (宽, 高)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* 天蓝色背景 */
                color: white;               /* 文字颜色 */
                border: none;               /* 去掉边框 */
                border-radius: 5px;         /* 圆角边框 */
                font-size: 30px;            /* 字体大小 */
            }
            QPushButton:hover {
                background-color: #00BFFF;  /* 鼠标悬停时深天蓝色背景 */
            }
        """)

        self.generate_button.clicked.connect(self.on_generate_button_clicked)
        button_layout.addWidget(self.generate_button)

        # 添加间距，将按钮推到上方
        spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        button_layout.addItem(spacer_top)

        # 右侧的图像显示区域
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1024, 1024)  # 设置固定的图像显示大小
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")  # 设置背景为白色

        # 初始化为白色的空图像
        blank_image = QImage(1024, 1024, QImage.Format_RGB32)
        blank_image.fill(Qt.white)
        self.image_label.setPixmap(QPixmap.fromImage(blank_image))
        right_layout.addWidget(self.image_label)

        # 将按钮布局添加到右侧布局
        right_layout.addLayout(button_layout)
        right_widget.setLayout(right_layout)

        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)

    def update_slider_value(self):
        # 更新滑动条当前值显示的 QLabel 文本
        current_value = self.slider.value()
        self.slider_value_label.setText(f"当前步数: {current_value}")

    def on_generate_button_clicked(self):
        # 获取用户输入的提示词
        prompt = self.prompt_edit.toPlainText().strip()
        a_prompt = self.a_prompt_edit.toPlainText().strip()
        n_prompt = self.n_prompt_edit.toPlainText().strip()
        name_ = self.name_label.toPlainText().strip()

        if not prompt:
            return  # 如果没有输入提示词，则返回

        # 获取滑动条的值
        ddim_steps = self.slider.value()

        # 调用 Stable Diffusion 模型生成图像
        x_samples = generate_image(prompt, a_prompt, n_prompt,ddim_steps,name_)
        for index, image in enumerate(x_samples):
            # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image= cv2.resize(image,(1024,1024), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            # print(index)
            # 显示生成的图像
            self.display_image(image)

        print("图像生成成功。")

    def display_image(self, image):
        try:
            pil_image = Image.fromarray(image)
            rgb_image = pil_image.convert('RGB')

            image_array = np.array(rgb_image)

            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            print("图像成功显示。")
        except Exception as e:
            print(f"显示图像时出错: {e}")


def main():
    # 创建应用程序
    app = QApplication(sys.argv)
    window = StableDiffusionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
