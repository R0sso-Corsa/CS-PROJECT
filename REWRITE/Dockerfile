#

#FROM tensorflow/tensorflow:2.13.0-gpu
#WORKDIR /app

#ENV TF_CPP_MIN_LOG_LEVEL=2

#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt && \
    #python - <<'PY'
#import pathlib, re
#candidates = [
#    pathlib.Path('/usr/local/lib/python3.8/dist-packages'),
#    pathlib.Path('/usr/local/lib/python3.9/dist-packages'),
#    pathlib.Path('/usr/local/lib/python3.8/site-packages'),
#]
#for base in candidates:
#    f = base / 'multitasking' / '__init__.py'
 #   if f.exists():
  #      txt = f.read_text()
   #     if 'from typing import Type' not in txt:
    #        txt = 'from typing import Type\n' + txt
     #   txt = re.sub(r'type\\[([^\\]]+)\\]', r'Type[\\1]', txt)
      #  f.write_text(txt)
      #  print('patched', f)
       # break
#else:
 #   print('multitasking __init__.py not found, skipping patch')
#PY

#COPY . /app

#RUN useradd -m -u 1000 appuser || true \
 #&& chown -R appuser:appuser /app

# USER appuser

#CMD ["python", "test.py"]

#FROM tensorflow/tensorflow:2.13.0-gpu
#WORKDIR /app
#ENV TF_CPP_MIN_LOG_LEVEL=2

#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

#COPY . /app
#RUN useradd -m -u 1000 appuser || true && chown -R appuser:appuser /app
#USER appuser
#CMD ["python", "test.py"]


#BUILD DOCKER IMAGE
#docker build -t gpu-test:latest "C:\Users\Paron\OneDrive\Desktop\testing"

#RUN DOCKER IMAGE {WITH GPU ACCESS}
#docker run --gpus all --rm -v "C:\Users\Paron\OneDrive\Desktop\testing:/app" gpu-test:latest


FROM rocm/tensorflow:latest

# Copy requirements.txt and install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy your main script
COPY future-Eric.py /workspace/future-Eric.py

# Set the default command
CMD ["python", "/workspace/future-Eric.py"]

