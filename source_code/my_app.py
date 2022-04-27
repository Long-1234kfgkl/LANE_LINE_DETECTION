import streamlit as st
import main
import tempfile
import cv2 as cv2
import time

def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

load_css('style.css')

with st.sidebar:
    t = "<span class='bold blue highlight'>Chọn hình ảnh hoặc video</span>"
    st.markdown(t, unsafe_allow_html=True)
    type_fg_file = st.selectbox("", ['image', 'video'])

    # if type_fg_file != 'webcam':
    if type_fg_file == 'video':
        text = 'video'
        type = 'mp4'
    else:
        text = 'image'
        type = ['png', 'jpg', 'jpeg']


    f = st.file_uploader('', type, key='12')
    btn = st.button('Run')
    if not btn:
        st.stop()
    fg = tempfile.NamedTemporaryFile(delete=False)
    fg.write(f.read())
    fg.close()


if text == 'video':
    vf = cv2.VideoCapture(fg.name)

    stframe = st.empty()

    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret :
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fra = main.lane_finding_pipeline(gray)
        stframe.image(fra)
        # time.sleep(0.01)
else:
    fg_ = cv2.imread(fg.name)
    fg_ = cv2.cvtColor(fg_, cv2.COLOR_BGR2RGB)
    st.image(fg_, width = 682)

    IMAG = main.lane_finding_pipeline(fg_)
    st.image(IMAG, width=682)