import imageio
import glob
import click

@click.command()
@click.argument('file_ext')
@click.option('--duration', default=0.5, type=float, help='frame duration')
def main(file_ext, duration):
    filenames = sorted(glob.glob('*' + file_ext))
    with imageio.get_writer('out_gif.gif', mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image) 

if __name__ == "__main__":
    main()