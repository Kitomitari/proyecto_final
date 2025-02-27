from flask import Flask, render_template, request, redirect, url_for, flash, seccion

app = Flask(__name__)
app.secret_key = 'clave_secreta'

#usuario simi