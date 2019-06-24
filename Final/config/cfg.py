import os
import string

version = '1.0'

number = string.digits
Alphabet = string.ascii_uppercase
alphabet = string.ascii_lowercase

# 图像大小
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
MAX_CAPTCHA = 4

gen_char_set = number + Alphabet  # + alphabet + '_'

CHAR_SET_LEN = len(gen_char_set)

home_root = os.path.abspath('../')  # home目录 请修改成自己实际位置
workspace = os.path.join(home_root, 'work', 'crack%s' % version)  # 用于工作的文件夹

xlsx_path = os.path.join(workspace, 'xlsx')

ckpt_path = os.path.join(workspace, 'ckpt')  # 模型数据保存文件夹
ckpt_name = 'captcha_cracker.ckpt'
save_ckpt = os.path.join(ckpt_path, ckpt_name)

# 输出日志 tensorboard监控的内容
tb_log_path = os.path.join(workspace, 'tb_logs')

if __name__ == '__main__':
    print(CHAR_SET_LEN)
    print(home_root)
    print(workspace)
    print(save_ckpt)
    print(tb_log_path)
