
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class plotAnimator():

    def __init__(self, data):
        self.data = data
        self.plt = plt

        self.frames = 300

        def update_lines(num, dataLines, lines):
            lineNumber = -1
            for line, data in zip(lines, dataLines):
                lineNumber += 1
                lineStep = num // (len(data[0])+1)
                lineIndex = num % (len(data[0])+1)
                if(lineStep == lineNumber):
                    # NOTE: there is no .set_data() for 3 dim data... to get first two array
                    line.set_data(data[0:2, :])
                    # line.set_3d_properties(data[2, :lineIndex])
                    # to get last z array
                    line.set_3d_properties(data[2])
                    line.set_alpha(.5)
            return lines

        # Attaching 3D axis to the figure
        fig = self.plt.figure()
        self.ax = p3.Axes3D(fig)

        # Creating fifty line objects.
        # NOTE: Can't pass empty arrays into 3d version of plot()
        lines = [self.ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0]
                 for dat in self.data]

        # Setting the axes properties
        self.ax.set_xlim3d([0.0, 12])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([0.0, 16])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([0.0, 5.5])
        self.ax.set_zlabel('Z')

        self.ax.set_title(
            'Matrix Factorization with Distributed Stochastic Gradient Descent')

        # Creating the Animation object
        self.line_ani = animation.FuncAnimation(
            fig, update_lines, self.frames, fargs=(data, lines), interval=50)

        # plt.rcParams['animation.ffmpeg_path'] = 'e:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
        # # writer = animation.FFMpegWriter(fps=30, codec='libx264')  # Or
        # # writer = animation.FFMpegWriter(
        # #     fps=20, metadata=dict(artist='mberneti'), bitrate=1800)
        # # Set up formatting for the movie files
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=30, metadata=dict(artist='mberneti'), bitrate=1800)
        # self.line_ani.save('im.mp4', writer=writer)
