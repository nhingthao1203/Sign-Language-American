# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter.font as tkFont
import time  # Thêm import time cho tính năng 3 giây

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model(r'E:\MSASL-valid-dataset-downloader\bestmodel.h5')
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        # Thêm các biến cho tính năng 3 giây nhận diện
        self.char_start_time = None
        self.stable_char = None
        self.stable_char_count = 0
        self.STABLE_TIME_THRESHOLD = 3.0  # 3 giây
        self.MIN_STABLE_COUNT = 15  # Số lần nhận diện tối thiểu (tương ứng khoảng 0.5 giây với 30fps)

        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        self.setup_gui()
        self.str = ""  # Bắt đầu với chuỗi rỗng thay vì " "
        self.ccc = 0
        self.word = ""
        self.current_symbol = "C"
        self.photo = "Empty"
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""

        self.update_sentence_box()
        self.video_loop()

    def update_sentence_box(self):
        # Cập nhật hiển thị câu
        self.panel5.config(text=self.str)
        self.root.after(100, self.update_sentence_box)  # Giảm xuống 100ms cho cập nhật mượt hơn

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("🤟 Nhận Dạng Ngôn Ngữ Ký Hiệu")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x800")
        self.root.configure(bg='#1e293b')

        # Tạo style cho ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Cấu hình màu sắc
        self.style.configure('Title.TLabel',
                             background='#1e293b',
                             foreground='#e2e8f0',
                             font=('Segoe UI', 28, 'bold'))

        self.style.configure('Header.TLabel',
                             background='#1e293b',
                             foreground='#94a3b8',
                             font=('Segoe UI', 14, 'bold'))

        self.style.configure('Content.TLabel',
                             background='#334155',
                             foreground='#f1f5f9',
                             font=('Segoe UI', 16),
                             relief='solid',
                             borderwidth=1)

        self.style.configure('Modern.TButton',
                             background='#3b82f6',
                             foreground='white',
                             font=('Segoe UI', 12, 'bold'),
                             borderwidth=0,
                             focuscolor='none')

        self.style.map('Modern.TButton',
                       background=[('active', '#2563eb'),
                                   ('pressed', '#1d4ed8')])

        # Header chính
        header_frame = tk.Frame(self.root, bg='#1e293b', height=80)
        header_frame.pack(fill='x', padx=20, pady=(10, 20))
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame,
                               text="🤟 Nhận Dạng Ngôn Ngữ Ký Hiệu",
                               bg='#1e293b',
                               fg='#e2e8f0',
                               font=('Segoe UI', 32, 'bold'))
        title_label.pack(expand=True)

        # Frame chính
        main_frame = tk.Frame(self.root, bg='#1e293b')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Frame trái - Video
        left_frame = tk.Frame(main_frame, bg='#334155', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Header chung cho cả frame trái
        main_header = tk.Label(left_frame,
                               text="📹 Camera & Ảnh Tham Khảo",
                               bg='#334155',
                               fg='#94a3b8',
                               font=('Segoe UI', 16, 'bold'))
        main_header.pack(pady=10)

        # Container chứa video và ảnh
        content_container = tk.Frame(left_frame, bg='#334155')
        content_container.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # Phần trái - Video
        video_section = tk.Frame(content_container, bg='#334155')
        video_section.pack(side='left', fill='both', expand=True, padx=(0, 5))

        video_label = tk.Label(video_section,
                               text="📹 Video",
                               bg='#334155',
                               fg='#94a3b8',
                               font=('Segoe UI', 12, 'bold'))
        video_label.pack(pady=(0, 5))

        self.panel = tk.Label(video_section, bg='#1e293b', relief='sunken', bd=2)
        self.panel.pack(fill='both', expand=True)

        # Phần phải - Ảnh
        image_section = tk.Frame(content_container, bg='#334155')
        image_section.pack(side='right', fill='both', expand=True, padx=(5, 0))

        image_label = tk.Label(image_section,
                               text="🖼️ Ảnh",
                               bg='#334155',
                               fg='#94a3b8',
                               font=('Segoe UI', 12, 'bold'))
        image_label.pack(pady=(0, 5))

        # Hiển thị ảnh có sẵn
        try:
            # Thay đổi đường dẫn ảnh của bạn ở đây
            image_path = r"E:\MSASL-valid-dataset-downloader\Untitled.png"  # Đường dẫn đến ảnh của bạn

            pil_image = Image.open(image_path)
            # Resize ảnh để vừa khung (vì chia ngang)
            pil_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_panel = tk.Label(image_section, image=photo, bg='#1e293b', relief='sunken', bd=2)
            self.image_panel.image = photo  # Giữ reference
            self.image_panel.pack(fill='both', expand=True)

        except Exception as e:
            # Nếu không load được ảnh, hiển thị placeholder
            self.image_panel = tk.Label(image_section,
                                        text="Không thể tải ảnh\nKiểm tra đường dẫn file",
                                        bg='#1e293b',
                                        fg='#64748b',
                                        font=('Segoe UI', 10),
                                        relief='sunken',
                                        bd=2)
            self.image_panel.pack(fill='both', expand=True)
        # Frame phải
        right_frame = tk.Frame(main_frame, bg='#1e293b')
        right_frame.pack(side='right', fill='both', padx=(10, 0))

        # Frame hiển thị hand tracking (tạo nhưng không hiển thị)
        hand_frame = tk.Frame(right_frame, bg='#334155', relief='raised', bd=2)
        # hand_frame.pack(fill='x', pady=(0, 15))  # Comment dòng này

        hand_header = tk.Label(hand_frame,
                               text="✋ Theo Dõi Tay",
                               bg='#334155',
                               fg='#94a3b8',
                               font=('Segoe UI', 16, 'bold'))
        hand_header.pack(pady=10)

        self.panel2 = tk.Label(hand_frame, bg='#1e293b', relief='sunken', bd=2)
        self.panel2.pack(padx=20, pady=(0, 20))

        # Frame kết quả
        result_frame = tk.Frame(right_frame, bg='#334155', relief='raised', bd=2)
        result_frame.pack(fill='x', pady=(0, 15))

        # Ký tự hiện tại với thanh tiến trình
        char_frame = tk.Frame(result_frame, bg='#334155')
        char_frame.pack(fill='x', padx=20, pady=10)

        char_label = tk.Label(char_frame,
                              text="Ký tự:",
                              bg='#334155',
                              fg='#94a3b8',
                              font=('Segoe UI', 14, 'bold'))
        char_label.pack(side='left')

        self.panel3 = tk.Label(char_frame,
                               text="",
                               bg='#1e293b',
                               fg='#22d3ee',
                               font=('Segoe UI', 24, 'bold'),
                               width=5,
                               relief='sunken',
                               bd=2)
        self.panel3.pack(side='right')

        # Thêm thanh tiến trình 3 giây
        progress_frame = tk.Frame(result_frame, bg='#334155')
        progress_frame.pack(fill='x', padx=20, pady=(0, 10))

        progress_label = tk.Label(progress_frame,
                                  text="Tiến trình 3s:",
                                  bg='#334155',
                                  fg='#94a3b8',
                                  font=('Segoe UI', 12, 'bold'))
        progress_label.pack(anchor='w')

        self.progress_bar = ttk.Progressbar(progress_frame,
                                            mode='determinate',
                                            length=300,
                                            maximum=100)
        self.progress_bar.pack(fill='x', pady=(5, 0))

        # Câu hoàn chỉnh
        sentence_frame = tk.Frame(result_frame, bg='#334155')
        sentence_frame.pack(fill='x', padx=20, pady=10)

        sentence_label = tk.Label(sentence_frame,
                                  text="Câu:",
                                  bg='#334155',
                                  fg='#94a3b8',
                                  font=('Segoe UI', 14, 'bold'))
        sentence_label.pack(anchor='w')

        self.panel5 = tk.Label(sentence_frame,
                               text="",
                               bg='#1e293b',
                               fg='#f1f5f9',
                               font=('Segoe UI', 14),
                               justify='left',
                               wraplength=400,
                               relief='sunken',
                               bd=2,
                               height=3)  # Tăng chiều cao để hiển thị nhiều text hơn
        self.panel5.pack(fill='x', pady=(5, 0))

        # Frame gợi ý từ
        suggest_frame = tk.Frame(right_frame, bg='#334155', relief='raised', bd=2)
        suggest_frame.pack(fill='x', pady=(0, 15))

        suggest_header = tk.Label(suggest_frame,
                                  text="💡 Gợi Ý Từ",
                                  bg='#334155',
                                  fg='#fbbf24',
                                  font=('Segoe UI', 16, 'bold'))
        suggest_header.pack(pady=10)

        # Buttons gợi ý
        buttons_frame = tk.Frame(suggest_frame, bg='#334155')
        buttons_frame.pack(padx=20, pady=(0, 20))

        self.b1 = tk.Button(buttons_frame,
                            text="",
                            bg='#3b82f6',
                            fg='white',
                            font=('Segoe UI', 11, 'bold'),
                            relief='flat',
                            bd=0,
                            padx=10,
                            pady=5,
                            command=self.action1)
        self.b1.pack(fill='x', pady=2)

        self.b2 = tk.Button(buttons_frame,
                            text="",
                            bg='#3b82f6',
                            fg='white',
                            font=('Segoe UI', 11, 'bold'),
                            relief='flat',
                            bd=0,
                            padx=10,
                            pady=5,
                            command=self.action2)
        self.b2.pack(fill='x', pady=2)

        self.b3 = tk.Button(buttons_frame,
                            text="",
                            bg='#3b82f6',
                            fg='white',
                            font=('Segoe UI', 11, 'bold'),
                            relief='flat',
                            bd=0,
                            padx=10,
                            pady=5,
                            command=self.action3)
        self.b3.pack(fill='x', pady=2)

        self.b4 = tk.Button(buttons_frame,
                            text="",
                            bg='#3b82f6',
                            fg='white',
                            font=('Segoe UI', 11, 'bold'),
                            relief='flat',
                            bd=0,
                            padx=10,
                            pady=5,
                            command=self.action4)
        self.b4.pack(fill='x', pady=2)

        # Frame điều khiển
        control_frame = tk.Frame(right_frame, bg='#334155', relief='raised', bd=2)
        control_frame.pack(fill='x')

        control_header = tk.Label(control_frame,
                                  text="🎛️ Điều Khiển",
                                  bg='#334155',
                                  fg='#94a3b8',
                                  font=('Segoe UI', 16, 'bold'))
        control_header.pack(pady=10)

        control_buttons = tk.Frame(control_frame, bg='#334155')
        control_buttons.pack(padx=20, pady=(0, 20))

        # Hàng đầu tiên - Speak và Clear
        first_row = tk.Frame(control_buttons, bg='#334155')
        first_row.pack(fill='x', pady=(0, 10))

        self.speak = tk.Button(first_row,
                               text="🔊 Phát âm",
                               bg='#10b981',
                               fg='white',
                               font=('Segoe UI', 12, 'bold'),
                               relief='flat',
                               bd=0,
                               padx=15,
                               pady=8,
                               command=self.speak_fun)
        self.speak.pack(side='left', padx=(0, 10), fill='x', expand=True)

        self.clear = tk.Button(first_row,
                               text="🗑️ Xóa ký tự",
                               bg='#ef4444',
                               fg='white',
                               font=('Segoe UI', 12, 'bold'),
                               relief='flat',
                               bd=0,
                               padx=15,
                               pady=8,
                               command=self.clear_fun)
        self.clear.pack(side='right', padx=(10, 0), fill='x', expand=True)

        # Hàng thứ hai - Space và Clear All
        second_row = tk.Frame(control_buttons, bg='#334155')
        second_row.pack(fill='x')

        self.space_btn = tk.Button(second_row,
                                   text="␣ Thêm khoảng trắng",
                                   bg='#6366f1',
                                   fg='white',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat',
                                   bd=0,
                                   padx=15,
                                   pady=8,
                                   command=self.add_space)
        self.space_btn.pack(side='left', padx=(0, 10), fill='x', expand=True)

        self.clear_all = tk.Button(second_row,
                                   text="🗑️ Xóa tất cả",
                                   bg='#dc2626',
                                   fg='white',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat',
                                   bd=0,
                                   padx=15,
                                   pady=8,
                                   command=self.clear_all_fun)
        self.clear_all.pack(side='right', padx=(10, 0), fill='x', expand=True)

        # Thêm hiệu ứng hover cho các button
        self.add_hover_effect(self.b1, '#2563eb', '#3b82f6')
        self.add_hover_effect(self.b2, '#2563eb', '#3b82f6')
        self.add_hover_effect(self.b3, '#2563eb', '#3b82f6')
        self.add_hover_effect(self.b4, '#2563eb', '#3b82f6')
        self.add_hover_effect(self.speak, '#059669', '#10b981')
        self.add_hover_effect(self.clear, '#dc2626', '#ef4444')
        self.add_hover_effect(self.space_btn, '#4f46e5', '#6366f1')
        self.add_hover_effect(self.clear_all, '#b91c1c', '#dc2626')

    def add_hover_effect(self, button, hover_color, normal_color):
        def on_enter(e):
            button.config(bg=hover_color)

        def on_leave(e):
            button.config(bg=normal_color)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def add_space(self):
        """Thêm khoảng trắng vào câu"""
        self.str += " "
        self.panel5.config(text=self.str)

    def clear_fun(self):
        """Xóa ký tự cuối cùng (từ phải sang trái)"""
        if len(self.str) > 0:
            self.str = self.str[:-1]
        self.panel5.config(text=self.str)
        self.update_word_suggestions()

    def clear_all_fun(self):
        """Xóa toàn bộ câu"""
        self.str = ""
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""
        self.panel5.config(text=self.str)
        self.update_buttons()

    def update_char_stability(self, detected_char):
        """Cập nhật trạng thái ổn định của ký tự được nhận diện"""
        current_time = time.time()

        # Nếu ký tự thay đổi, reset
        if detected_char != self.stable_char:
            self.stable_char = detected_char
            self.stable_char_count = 1
            self.char_start_time = current_time
            self.progress_bar['value'] = 0
        else:
            # Ký tự giống nhau, tăng đếm
            self.stable_char_count += 1

            if self.char_start_time:
                elapsed_time = current_time - self.char_start_time
                progress = min((elapsed_time / self.STABLE_TIME_THRESHOLD) * 100, 100)
                self.progress_bar['value'] = progress

                # Nếu đã đủ thời gian và số lần nhận diện, tự động thêm ký tự
                if (elapsed_time >= self.STABLE_TIME_THRESHOLD and
                        self.stable_char_count >= self.MIN_STABLE_COUNT and
                        detected_char not in [' ', 'next', 'Backspace', None, ''] and
                        detected_char.isalnum()):
                    self.add_char_to_sentence(detected_char)
                    # Reset sau khi thêm
                    self.reset_char_stability()

    def reset_char_stability(self):
        """Reset trạng thái ổn định"""
        self.stable_char = None
        self.stable_char_count = 0
        self.char_start_time = None
        self.progress_bar['value'] = 0

    def add_char_to_sentence(self, char):
        """Thêm ký tự vào câu"""
        if char and char.strip():
            self.str += char.upper()
            self.update_word_suggestions()
            print(f"Đã thêm ký tự: {char}")

    def update_word_suggestions(self):
        """Cập nhật gợi ý từ"""
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st + 1:ed]
            self.word = word

            if len(word.strip()) != 0:
                suggestions = ddd.suggest(word.lower())
                lenn = len(suggestions)

                self.word1 = suggestions[0] if lenn >= 1 else ""
                self.word2 = suggestions[1] if lenn >= 2 else ""
                self.word3 = suggestions[2] if lenn >= 3 else ""
                self.word4 = suggestions[3] if lenn >= 4 else ""
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = ""
        else:
            self.word1 = self.word2 = self.word3 = self.word4 = ""

        self.update_buttons()

    def update_buttons(self):
        """Cập nhật text cho các button gợi ý"""
        self.b1.config(text=self.word1 if self.word1.strip() else "Không có gợi ý")
        self.b2.config(text=self.word2 if self.word2.strip() else "Không có gợi ý")
        self.b3.config(text=self.word3 if self.word3.strip() else "Không có gợi ý")
        self.b4.config(text=self.word4 if self.word4.strip() else "Không có gợi ý")

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            if cv2image.any():
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                if hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    x, y, w, h = map['bbox']
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    white = cv2.imread(r"E:\MSASL-valid-dataset-downloader\white.jpg")
                    if image.any():
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz[0]:
                            hand = handz[0]
                            handmap = hand[0]
                            self.pts = handmap['lmList']

                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15

                            # Vẽ các đường nối hand landmarks
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 255, 0), 3)

                            # Vẽ các đường nối chính
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1),
                                     (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1),
                                     (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1),
                                     (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1),
                                     (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1),
                                     (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)

                            # Vẽ các điểm landmarks
                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 3, (0, 0, 255), -1)

                            res = white
                            self.predict(res)

                            self.current_image2 = Image.fromarray(res)
                            # Resize image để fit vào panel
                            self.current_image2 = self.current_image2.resize((350, 350), Image.Resampling.LANCZOS)
                            imgtk = ImageTk.PhotoImage(image=self.current_image2)

                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk)

                            self.panel3.config(text=self.current_symbol)
                            self.panel5.config(text=self.str)

                            # Cập nhật các button gợi ý
                            self.b1.config(text=self.word1 if self.word1.strip() else "Không có gợi ý")
                            self.b2.config(text=self.word2 if self.word2.strip() else "Không có gợi ý")
                            self.b3.config(text=self.word3 if self.word3.strip() else "Không có gợi ý")
                            self.b4.config(text=self.word4 if self.word4.strip() else "Không có gợi ý")

        except Exception as e:
            print(f"Lỗi trong video_loop: {e}")
            print(traceback.format_exc())
        finally:
            self.root.after(30, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        if self.word1.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word1.upper()

    def action2(self):
        if self.word2.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word2.upper()

    def action3(self):
        if self.word3.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word3.upper()

    def action4(self):
        if self.word4.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word4.upper()

    def speak_fun(self):
        if self.str.strip():
            self.speak_engine.say(self.str)
            self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0
        self.update_char_stability(self.current_symbol)
        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

    def destructor(self):
        print("Đang đóng ứng dụng...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("🚀 Khởi động ứng dụng nhận dạng ngôn ngữ ký hiệu...")
    try:
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"Lỗi khởi động ứng dụng: {e}")
        print(traceback.format_exc())