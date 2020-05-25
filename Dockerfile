FROM tensorflow/tensorflow:2.2.0-gpu

RUN mkdir -p /opt/4P
COPY requirements_tf.txt /opt/4P/requirements_tf.txt
RUN pip install -r /opt/4P/requirements_tf.txt
COPY . /opt/4P
WORKDIR /opt/4P
RUN python common/client_data.py

CMD python tflib/train.py --flagfile tflib/config/gpu0.cfg --experiment_name gpu0