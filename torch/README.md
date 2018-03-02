luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec

pip install  easydict

git clone git@github.com:Element-Research/rnn.git
cd rnn
luarocks make rocks/rnn-scm-1.rockspec

luarocks install utf8
