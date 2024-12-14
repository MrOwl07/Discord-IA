import discord, os , random as r, requests, IA_def as IA, os
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()


TOKEN = os.getenv('dt')
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'Hemos iniciado sesión como {bot.user}')

def get_duck_image_url():
    url = 'https://random-d.uk/api/random'
    res = requests.get(url)
    data = res.json()
    return data['url']


@bot.command('duck')
async def duck(ctx):
    '''El comando "duck" devuelve la foto del pato'''
    print('hello')
    image_url = get_duck_image_url()
    await ctx.send(image_url)


@bot.command(name='IA')
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            
            # Guardar imagen localmente
            local_path = f"./{attachment.filename}"
            await attachment.save(local_path)
            
            # Obtener resultados de predicción
            class_name, confidence_score = IA.get_class(
                model_path="./keras_model.h5", 
                labels_path="labels.txt", 
                image_path=local_path
            )
            
            # Formatear la confianza como porcentaje
            confidence_percentage = confidence_score * 100

            # Crear un embed de Discord para la respuesta
            embed = discord.Embed(
                title="Resultado del análisis",
                description=f"**Clase predicha:** {class_name.strip()}\n**Confianza:** {confidence_percentage:.2f}%",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=file_url)
            embed.set_footer(text="Análisis realizado con Keras y TensorFlow")

            # Enviar el embed al canal
            await ctx.send(embed=embed)
            
            # Eliminar archivo local
            os.remove(local_path)
    else:
        await ctx.send("Olvidaste subir la imagen :(")


bot.run(TOKEN)