#!/bin/sh

cd ..


train_particle=ntau_10GeV_10    # ntau_10GeV_10      uds91
test_particle=ntau_10GeV_10     # ntau_10GeV_10      uds91



python save_root.py /home/murata/data_murata/code_test/test test/checkpoint/ckpt_0_1.pth.tar test/test.root 0 500000 False 7 6 False

