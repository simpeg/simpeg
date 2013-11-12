from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from matplotlib import animation

# http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/
# http://www.renevolution.com/how-to-install-ffmpeg-on-mac-os-x/

VIDEO_TAG = """<video controls loop>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
    plt.close(anim._fig)
    return anim_to_html(anim)

animation.Animation._repr_html_ = display_animation

easyAnimate = animation.FuncAnimation
