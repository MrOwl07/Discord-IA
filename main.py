import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
import IA_def as IA

load_dotenv()

TOKEN = os.getenv('dt')
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='-', intents=intents)

@bot.event
async def on_ready():
    print(f'Hemos iniciado sesión como {bot.user}')

@bot.command(name='IA')
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            try:
                # Guardar imagen localmente
                local_path = f"./{attachment.filename}"
                await attachment.save(local_path)

                # Detectar objetos en la imagen
                detected_objects, processed_image_path = IA.detect_objects(local_path)

                if not detected_objects:
                    await ctx.send("Lo siento, no estoy seguro de lo que se muestra en la imagen.")
                    os.remove(local_path)
                    return
                
                # Crear mensaje con los objetos detectados
                objects = ", ".join(detected_objects)
                await ctx.send(f"He detectado los siguientes objetos en la imagen: {objects}")

                # Verificar si se detecta un ave
                if any(obj in ['bird', 'pigeon', 'sparrow'] for obj in detected_objects):
                    recommendation = (
                        "Esto es un ave. Puedes alimentarla con semillas como girasol, maíz o avena. "
                        "Recuerda ofrecer agua fresca y un espacio seguro."
                    )
                    await ctx.send(recommendation)

                # Clasificar la imagen con Keras
                class_name, confidence_score, all_predictions = IA.classify_image(local_path)
                confidence_percentage = confidence_score * 100
                
                # Mostrar todas las predicciones en consola
                print(f"Predicciones detalladas: {all_predictions}")

                # Crear un embed para el mensaje
                embed = discord.Embed(
                    title="Resultado del análisis",
                    description=f"**Clase predicha:** {class_name}\n**Confianza:** {confidence_percentage:.2f}%",
                    color=discord.Color.blue()
                )
                embed.set_image(url=f"attachment://{processed_image_path}")
                embed.set_footer(text="Análisis realizado con Keras y TensorFlow")
                
                # Enviar el embed con la imagen procesada
                with open(processed_image_path, 'rb') as img_file:
                    discord_file = discord.File(img_file, filename="detected_image.jpg")
                    await ctx.send(embed=embed, file=discord_file)

                # Eliminar archivos temporales
                os.remove(local_path)
                os.remove(processed_image_path)

            except Exception as e:
                await ctx.send("Ocurrió un error al procesar la imagen. Por favor, intenta nuevamente.")
                print(f"Error: {e}")
    else:
        await ctx.send("Olvidaste subir la imagen :(")

bot.run(TOKEN)
