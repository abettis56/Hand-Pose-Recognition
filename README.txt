README

So, a brief explanation of where I left off and where I was headed with this:

My plan was to take input from the Leap and restructure it so that each proximal phalanx was recorded relative to the hand's basis vectors. Each intermediate and distal phalanx is then recorded using the basis of the previous bone. In this manner, each bone will be rotationally and positionally invariant, and all data for each relevant joint will be preserved, as the only things that matter when recording the position of a joint is the relative rotations of the involved bones.

I believe I have a system in "basis_recorder.py" that implements this correctly.

Unfortunately, testing it completely requires a visualizer program. Since I'm familiar with THREE.js, I took the modified index.js program I had access to and used that as a visualizer. However, reversing the above-described normalization procedure is required to accurately display a hand. Since the procedure was conducted with numpy, I felt it best to undo it in numpy as well, when resulted in "display_converter.py", a python script that simply reverses the procedure performed in "basis_recorder.py". I was nearly complete with the visualizer when I needed to switch focus away from this project.

As described in the header comments in the 3 relevant programs, any problems would most likely lie in "index.js" or "display_converter.py". Of course, it's also possible that I simply overlooked something in "basis_recorder.py".

The only programs that currently contain working code are:
- basis_recorder.py
- display_converter.py
- index.js

Moving forward, the plan was to come up with a way to use this system to generate specific error poses (which would have been tremendously difficult without any form of normalization) and adjust the autoencoder to accommodate 135-dimensional vectors.