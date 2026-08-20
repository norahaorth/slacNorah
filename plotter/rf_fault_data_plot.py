## This is old plotting file from DESY / Sonya.
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.messagebox import showwarning
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path
import datetime
import argparse
import base64
import io
import re
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler, cbook
from matplotlib.figure import Figure
import matplotlib
import numpy as np
from scipy import signal

pyperclip_imported = False

try:
    import pyperclip

    pyperclip_imported = True
except ImportError:
    pyperclip_imported = False

rescale_image = (
    b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsM"
    b"AAA7DAcdvqGQAAAFRSURBVGhD7ZeBrsMgCEW1///PTpq6KLFWFLBsnqQvS+zgXq+yPO8uvPfh+lgQQjjfqa231oCZ9d7vHvDn7q"
    b"U3kzQfFsUnQPuZgGWiiTKBdLbeCtZrPoFtYPPvmJ9CqgZwrztaGnCNPYUoSKSrngC3icKA1gWe6YO/qyIY83SZKQa/L/ZOiBFyQ"
    b"T19WgaqU2ileMpu1xD9h6ZX/IwJsSlE3flRE7F2WXxmN2r0iKeA64n+DnCLryGWwJN4WOfoJZJAj/jr4zSxFm8CFPEjvXB91gQo"
    b"4rkgGQABdyKexEsxlAAWu0o8MHyEkuiV4oGpO7BaPMB6iTXEQ4/8YTOgIT4ByaeHzUDtOEmA+7AeIS0TOawGAG0T7Aa0iRtW7pj"
    b"kZeTohWuYM4Axf4TMGYAU82cnsJrfG6OapClU09Bay7F/hCTnfguOvlDjTEDbBJd455z7APTz9WNZa4ZSAAAAAElFTkSuQmCC"
)

popup_image = (
    b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsM"
    b"AAA7DAcdvqGQAAADGSURBVGhD7dfrCsIwDAXg6vs/sTDUQQKxnJVuzZoUz/cniCwmvc5CRER/7SHRekucCdXR5SlxWWwg2vINIP"
    b"smnrWRh3+LSygaG4jGBqIt3wAy8x4Y1nobvfyG+HXnAPzUxT0QjQ1k5HEKeeSowZzLz4D3n3rNpzlGjuIazJlpBuASiXKlGH2mf"
    b"k4/o+9SzYBdGrbopmybGDXRlPEUOrXxMzbQNfIqWwOo+JfEqfZCTo1kB5jT86KxvIu30l5kRzaJRERElVI+zs8oDue1LXkAAAAA"
    b"SUVORK5CYII="
)

types_map = {
    "ACQ_FLT_TS": "ts1",
    "SSA:CALTS": "ts2",
    "PROBECALTS": "ts2",
    "CAV:FLTAWF": "array",
    "CAV:FLTPWF": "array",
    "CAV:FLTPWRWF": "array",
    "CAV:FLTIWF": "array",
    "CAV:FLTQWF": "array",
    "CAV:FLTTWF": "array",
    "FWD:FLTAWF": "array",
    "FWD:FLTPWF": "array",
    "FWD:FLTPWRWF": "array",
    "FWD:FLTIWF": "array",
    "FWD:FLTQWF": "array",
    "FWD:FLTTWF": "array",
    "REV:FLTAWF": "array",
    "REV:FLTPWF": "array",
    "REV:FLTPWRWF": "array",
    "REV:FLTIWF": "array",
    "REV:FLTQWF": "array",
    "REV:FLTTWF": "array",
    "CTRL:FLTLIMS_IL": "array",
    "CTRL:FLTLIMS_IH": "array",
    "CTRL:FLTLIMS_QL": "array",
    "CTRL:FLTLIMS_QH": "array",
    "DECAYREFWF": "array",
    "PWRDISSWF": "array",
}


class LogContent:
    def __init__(self, log_title, filename, data):
        self.log_title = log_title
        self.filename = filename
        self.data = data


class ParserResult:
    signal_name = None
    value = None


def parse_comment(line):
    elements = line.split(" ")
    if elements[0] != "#":
        return None
    signal_prefix = elements[1]
    plot_title = " ".join(elements[2:])
    return [signal_prefix, plot_title]


def parse_line(line, signal_prefix, line_number):
    elements = line.split(" ")
    result = ParserResult()

    if not elements[0].startswith(signal_prefix):
        showwarning(
            title="Malformed file",
            message=f"Line {line_number} does not start with the cavity name.",
        )
        return None

    result.signal_name = elements[0][len(signal_prefix) :]

    # some old files could contain values for limits that are very long arrays
    # this if condition makes it support the single value while also keeping backward compatibility
    if (
        result.signal_name == "CTRL:FLTLIMS_IL"
        or result.signal_name == "CTRL:FLTLIMS_IH"
        or result.signal_name == "CTRL:FLTLIMS_QL"
        or result.signal_name == "CTRL:FLTLIMS_QH"
    ) and len(elements) == 3:
        result.value = [float(elements[2])]  # type: ignore
    elif types_map.get(result.signal_name) == "array":
        if len(elements) < 10:
            showwarning(
                title="Malformed file",
                message=f"{elements[0]} should be an array but contains only {len(elements) - 2} values.",
            )
            result.value = None
        else:
            values = np.array(elements[2:], dtype=np.float32)
            result.value = values
    elif types_map.get(result.signal_name) == "ts1":
        if len(elements) != 6:
            showwarning(
                title="Malformed file",
                message=f"{elements[0]} should be a date of type %b %d %Y %H:%M:%S.",
            )
            result.value = None
        else:
            date = " ".join(elements[2:]).strip()
            try:
                date_obj = datetime.datetime.strptime(date, "%b %d %Y %H:%M:%S")
            except ValueError:
                showwarning(
                    title="Malformed file",
                    message=f"{elements[0]} should be a date of type %b %d %Y %H:%M:%S.",
                )
                date_obj = None
            result.value = date_obj  # type: ignore
    elif types_map.get(result.signal_name) == "ts2":
        if len(elements) < 3:
            result.value = None
        else:
            date = elements[2]
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d-%H:%M:%S")
            except ValueError:
                showwarning(
                    title="Malformed file",
                    message=f"{elements[0]} should be a date of type %Y-%m-%d-%H:%M:%S.",
                )
                date_obj = None
            result.value = date_obj  # type: ignore
    else:
        if len(elements) == 3:
            result.value = float(elements[2])  # type: ignore
        elif len(elements) == 2:
            # no value available
            result.value = None
        else:
            # unknown value
            result.value = None
    return result


def parse_file(filename):
    data = {}

    with open(filename, "r") as file:
        line = file.readline().strip()
        result = parse_comment(line)
        if result is None:
            showwarning(
                title="File not supported", message=f"{filename} is not supported."
            )
            return None

        signal_prefix = result[0]
        log_title = f"{result[0]} {result[1]}"

        line_number = 1
        while True:
            line = file.readline().strip()
            line_number = line_number + 1
            if not line:
                break

            result = parse_line(line, signal_prefix, line_number)
            if result is not None:
                data[result.signal_name] = result.value
            else:
                showwarning(
                    title="File not supported", message=f"{filename} is not supported."
                )
                return None

    x_axis1 = data.get("CAV:FLTTWF")
    x_axis2 = data.get("FWD:FLTTWF")
    x_axis3 = data.get("REV:FLTTWF")
    if not (np.array_equal(x_axis1, x_axis2) and np.array_equal(x_axis2, x_axis3)):  # type: ignore
        showwarning(
            title="Malformed file",
            message="X axis is not equal for cavity, forward and reverse signals.",
        )
        return None

    sanitize_data(data, signal_prefix)

    calculate_quench_related_signals(data)

    return LogContent(log_title=log_title, filename=filename, data=data)


def sanitize_data(data, signal_prefix):
    elements = [
        "CTRL:FLTLIMS_IL",
        "CTRL:FLTLIMS_IH",
        "CTRL:FLTLIMS_QL",
        "CTRL:FLTLIMS_QH",
    ]

    for element in elements:
        if data.get(element) is not None:
            if len(data["CAV:FLTTWF"]) < len(data[element]):
                data[element] = data[element][: len(data["CAV:FLTTWF"])]
            elif len(data["CAV:FLTTWF"]) < len(data[element]):
                missing_elements = len(data["CAV:FLTTWF"]) - len(data[element])
                missing_elements_list = data[element][-1] * missing_elements
                data[element].extend(missing_elements_list)

    if data.get("FREQ") is None or data.get("IMPED") is None:
        data["FREQ"] = 1300e6
        data["IMPED"] = 1024  # Ohms
        if re.match(r"ACCL:L\dB:H\d{3}:", signal_prefix):
            data["FREQ"] = 3900e6
            data["IMPED"] = 750  # Ohms


def calculate_quench_related_signals(data):
    fsamp = data["FREQ"] / 14.0
    dt = 1.0 / fsamp * 33 * data.get("ACQ_DECIM") * 2

    cav = data["CAV:FLTIWF"] + 1j * data["CAV:FLTQWF"]
    fwd = data["FWD:FLTIWF"] + 1j * data["FWD:FLTQWF"]
    rev = data["REV:FLTIWF"] + 1j * data["REV:FLTQWF"]
    npt = len(cav)
    nstart = 0
    nfinish = npt
    nsearch = range(nstart, nfinish)
    # cdiff = np.diff(abs(cav)[nsearch])
    # q = np.argmax(abs(cdiff)) + nstart  # point of interest
    # print("POI n = %d  t = %.4f s" % (q, q * dt))

    ntd = np.array(nsearch)

    nt = ntd[4:-2]  # region of interest
    # t = dt * nt

    # result of deriv_fir(h_order=2, maxf=0.19)
    # has 1% DC error, is based on signal.remez()
    deriv_fira = [-0.1115, 0.7182, 0, -0.7182, 0.1115]

    fwd_p = abs(fwd)[nt] ** 2
    rev_p = abs(rev)[nt] ** 2
    w0 = 2 * np.pi * data["FREQ"]  # /s
    cav_u = abs(cav * 1e6)[ntd] ** 2 / (w0 * data["IMPED"])
    # cav_p = numpy.diff(cav_u) / dt
    # cav_p = numpy.append(cav_p, [cav_p[-1]])
    cav_p = signal.lfilter(deriv_fira, [1], cav_u) / dt
    cav_p = cav_p[6:]
    # cav_u = abs(cav*1e6)[nt]**2/(w0*RoverQ)

    data["QUENCH:FLTTWF"] = data["CAV:FLTTWF"][4:-2]
    data["QUENCH:CAVITY"] = -cav_p
    # result["cavity"] = signal.savgol_filter(-cav_p, 5, 2)
    data["QUENCH:SYSTEM"] = fwd_p - rev_p - cav_p
    data["QUENCH:WAVEGUIDE"] = fwd_p - rev_p
    if data.get("QUENCH_THRESH") is not None:
        data["QUENCH:THRESHOLD"] = (
            np.ones(len(data["QUENCH:FLTTWF"])) * data["QUENCH_THRESH"]
        )


class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse goes on it.

    see:

    http://stackoverflow.com/questions/3221956/
           what-is-the-simplest-way-to-make-tooltips-
           in-tkinter/36221216#36221216

    http://www.daniweb.com/programming/software-development/
           code/484591/a-tooltip-class-for-tkinter

    - Originally written by vegaseat on 2014.09.09.

    - Modified to include a delay time by Victor Zaccardo on 2016.03.25.

    - Modified
        - to correct extreme right and extreme bottom behavior,
        - to stay inside the screen whenever the tooltip might go out on
          the top but still the screen is higher than the tooltip,
        - to use the more flexible mouse positioning,
        - to add customizable background color, padding, waittime and
          wraplength on creation
      by Alberto Vassena on 2016.11.05.

      Tested on Ubuntu 16.04/16.10, running Python 3.5.2

    TODO: themes styles support
    """

    def __init__(
        self,
        widget,
        *,
        bg="#FFFFEA",
        pad=(5, 3, 5, 3),
        text="widget info",
        waittime=400,
        wraplength=250,
    ):
        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.onEnter)
        self.widget.bind("<Leave>", self.onLeave)
        self.widget.bind("<ButtonPress>", self.onLeave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None

    def onEnter(self, event=None):
        self.schedule()

    def onLeave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        def tip_pos_calculator(widget, label, *, tip_delta=(10, 5), pad=(5, 3, 5, 3)):
            w = widget

            width, height = (
                pad[0] + label.winfo_reqwidth() + pad[2],
                pad[1] + label.winfo_reqheight() + pad[3],
            )

            mouse_x, mouse_y = w.winfo_pointerxy()

            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]

            x_delta = x1 + width - w.winfo_screenwidth()
            x_delta = max(0, x_delta)
            y_delta = y1 + height - w.winfo_screenheight()
            y_delta = max(0, y_delta)

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:
                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height

            offscreen_again = y1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.

                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y1 = 0

            return x1, y1

        bg = self.bg
        pad = self.pad
        widget = self.widget

        # creates a toplevel window
        self.tw = tk.Toplevel(widget)

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)

        win = tk.Frame(self.tw, background=bg, borderwidth=0)
        label = tk.Label(
            win,
            text=self.text,
            justify=tk.LEFT,
            background=bg,
            relief=tk.SOLID,
            borderwidth=0,
            wraplength=self.wraplength,
        )

        label.grid(padx=(pad[0], pad[2]), pady=(pad[1], pad[3]), sticky=tk.NSEW)
        win.grid()

        x, y = tip_pos_calculator(widget, label)

        self.tw.wm_geometry("+%d+%d" % (x, y))

    def hide(self):
        tw = self.tw
        if tw:
            tw.destroy()
        self.tw = None


class NavigationToolbar(NavigationToolbar2Tk):
    home_scaling_factors = None
    ui_elements = None
    signal_name = None
    log_content = None

    def __init__(
        self,
        canvas,
        window,
        *,
        pack_toolbar=True,
        vertical_toolbar=False,
        background_color=None,
        popup_button=True,
    ):
        # Avoid using self.window (prefer self.canvas.get_tk_widget().master),
        # so that Tool implementations can reuse the methods.
        self.master = window
        self.vertical_toolbar = vertical_toolbar

        tk.Frame.__init__(
            self,
            master=window,
            borderwidth=2,
            width=int(canvas.figure.bbox.width),
            height=50,
            bg=background_color,
        )  # type: ignore

        self._buttons = {}

        self.rescale_figure = lambda: (
            rescale_figure(
                canvas.figure.axes, self.log_content, self.ui_elements, self.signal_name
            ),
            canvas.draw(),
        )
        self.home = lambda: (
            reset_figure_scale(canvas.figure.axes, self.home_scaling_factors),
            canvas.draw(),
        )  # type: ignore
        self.popup_figure = lambda: create_popup_figure(
            self.ui_elements,
            canvas.figure,
            self.log_content,
            self.signal_name,
            self.home_scaling_factors,
        )

        self.new_toolitems = []
        for item in self.toolitems:
            if item.count("Subplots") == 1:
                continue
            self.new_toolitems.append(item + (None,))
            if item.count("Forward") == 1:
                self.new_toolitems.append(
                    (
                        "Rescale",
                        "Rescale to the visible lines",
                        "home",
                        "rescale_figure",
                        rescale_image,
                    )
                )
            if item.count("Save") == 1 and popup_button:
                self.new_toolitems.append(
                    (
                        "Popup",
                        "Create a new window for this plot",
                        "home",
                        "popup_figure",
                        popup_image,
                    )
                )

        for (
            text,
            tooltip_text,
            image_file,
            callback,
            image_string,
        ) in self.new_toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(
                    text,
                    str(cbook._get_data_path(f"images/{image_file}.png")),
                    toggle=callback in ["zoom", "pan"],
                    command=getattr(self, callback),
                    image_string=image_string,
                )
                if tooltip_text is not None:
                    Tooltip(button, text=tooltip_text)

        self._label_font = tkfont.Font(root=window, size=10)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        label = tk.Label(
            master=self,
            font=self._label_font,
            text="\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}",
            bg=background_color,
        )  # type: ignore
        label.pack(side=tk.RIGHT)

        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(
            master=self,
            font=self._label_font,
            textvariable=self.message,
            bg=background_color,
        )  # type: ignore
        self._message_label.pack(side=tk.RIGHT)

        matplotlib.backend_bases.NavigationToolbar2.__init__(self, canvas)  # type: ignore
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)

    # override _Button() to re-pack the toolbar button in vertical direction
    def _Button(self, text, image_file, toggle, command, image_string=None):
        b = super()._Button(text, image_file, toggle, command)
        if image_string is not None:
            im = Image.open(io.BytesIO(base64.b64decode(image_string)))
            size = b.winfo_pixels("18p")
            image = ImageTk.PhotoImage(im.resize((size, size)), master=self)
            b.configure(image=image, height="18p", width="18p")
            b._ntimage = image  # type: ignore
        if self.vertical_toolbar:
            b.pack(side=tk.TOP)  # re-pack button in vertical direction
        return b

    # override _Spacer() to create vertical separator
    def _Spacer(self):
        if self.vertical_toolbar:
            s = tk.Frame(self, width=26, relief=tk.RIDGE, bg="DarkGray", padx=2)
            s.pack(side=tk.TOP, pady=5)  # pack in vertical direction
        else:
            s = tk.Frame(master=self, height="18p", relief=tk.RIDGE, bg="DarkGray")
            s.pack(side=tk.LEFT, padx="3p")
        return s

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass


class FigureObject:
    def __init__(
        self,
        line_cav,
        line_fwd,
        line_rev,
        canvas,
        toolbar,
        ax,
        ax2=None,
        line_l=None,
        line_h=None,
        line_decay_ref=None,
        line_cavity_du=None,
        line_system=None,
        line_waveguide=None,
        line_threshold=None,
    ):
        self.line_cav = line_cav
        self.line_fwd = line_fwd
        self.line_rev = line_rev
        self.canvas = canvas
        self.toolbar = toolbar
        self.ax = ax
        self.ax2 = ax2
        self.line_l = line_l
        self.line_h = line_h
        self.line_decay_ref = line_decay_ref
        self.line_cavity_du = line_cavity_du
        self.line_system = line_system
        self.line_waveguide = line_waveguide
        self.line_threshold = line_threshold


class UIElements:
    def __init__(
        self,
        root,
        title_sv,
        date_sv,
        ticks_sv,
        figures,
        path_text_sv,
        cavity_checkbox_value,
        forward_checkbox_value,
        reverse_checkbox_value,
        decay_checkbox_value,
        cavity_du_checkbox_value,
        system_checkbox_value,
        waveguide_checkbox_value,
        background_color,
    ):
        self.root = root
        self.title_sv = title_sv
        self.date_sv = date_sv
        self.ticks_sv = ticks_sv
        self.figures = figures
        self.path_text_sv = path_text_sv
        self.cavity_checkbox_value = cavity_checkbox_value
        self.forward_checkbox_value = forward_checkbox_value
        self.reverse_checkbox_value = reverse_checkbox_value
        self.decay_checkbox_value = decay_checkbox_value
        self.cavity_du_checkbox_value = cavity_du_checkbox_value
        self.system_checkbox_value = system_checkbox_value
        self.waveguide_checkbox_value = waveguide_checkbox_value
        self.parameters_button = None
        self.classification_button = None
        self.background_color = background_color


class ScalingFactor:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_limits(values):
    min_values = []
    max_values = []

    for elements in values:
        min_values.append(np.min(elements))
        max_values.append(np.max(elements))

    min_value = np.min(min_values)
    min_value = np.min([min_value, 0])
    max_value = np.max(max_values)
    interval = max_value - min_value
    min_value = min_value - (5 * interval / 100)
    max_value = max_value + (5 * interval / 100)

    return [min_value, max_value]


def calculate_scaling_factors(x_axis, y_axes1, y_axes2=None):
    interval = np.max(x_axis) - np.min(x_axis)
    x_min = np.min(x_axis) - (interval * 0.05)
    x_max = np.max(x_axis) + (interval * 0.05)
    x = [x_min, x_max]

    if y_axes2:
        data_limit1 = get_limits(y_axes1)
        data_limit2 = get_limits(y_axes2)

        y_lims = np.array([data_limit1, data_limit2])

        # normalize all axes
        y_mags = (y_lims[:, 1] - y_lims[:, 0]).reshape(len(y_lims), 1)
        y_lims_normalized = y_lims / y_mags

        # find combined range
        y_new_lims_normalized = np.array(
            [np.min(y_lims_normalized), np.max(y_lims_normalized)]
        )

        # denormalize combined range to get new axes
        new_lims = y_new_lims_normalized * y_mags

        y_min = new_lims[:, 0]
        y_max = new_lims[:, 1]
        y1 = [y_min[0], y_max[0]]
        y2 = [y_min[1], y_max[1]]
        return [ScalingFactor(x, y1), ScalingFactor(x, y2)]
    else:
        y_limits = get_limits(y_axes1)
        y = [y_limits[0], y_limits[1]]
        return [ScalingFactor(x, y)]


def reset_figure_scale(axes, scaling_factors):
    assert len(axes) == len(scaling_factors)

    axes[0].set_xlim(scaling_factors[0].x)
    axes[0].set_ylim(scaling_factors[0].y)
    if len(axes) > 1:
        axes[1].set_ylim(scaling_factors[1].y)


def rescale_figure(axes, log_content, ui_elements, signal_name):
    data = log_content.data

    substitutions = {
        "amplitude": "A",
        "phase": "P",
        "power": "PWR",
        "i": "I",
        "q": "Q",
        "quench": "",
    }

    dictionary_signal_name = f"FLT{substitutions[signal_name]}WF"

    y_axis1_list = []
    y_axis2_list = []

    if signal_name == "quench":
        x_axis = data["QUENCH:FLTTWF"]
        if ui_elements.cavity_du_checkbox_value.get() == 1:
            y_axis1_list.append(data["QUENCH:CAVITY"])
        if ui_elements.system_checkbox_value.get() == 1:
            y_axis1_list.append(data["QUENCH:SYSTEM"])
        if ui_elements.waveguide_checkbox_value.get() == 1:
            y_axis1_list.append(data["QUENCH:WAVEGUIDE"])
    else:
        x_axis = data["CAV:FLTTWF"]

        if ui_elements.cavity_checkbox_value.get() == 1:
            y_axis1_list.append(data["CAV:" + dictionary_signal_name])
        if ui_elements.decay_checkbox_value.get() == 1 and signal_name == "amplitude":
            y_axis1_list.append(data.get("DECAYREFWF"))

        if len(axes) > 1:
            if ui_elements.forward_checkbox_value.get() == 1:
                y_axis2_list.append(data["FWD:" + dictionary_signal_name])
            if ui_elements.reverse_checkbox_value.get() == 1:
                y_axis2_list.append(data["REV:" + dictionary_signal_name])
        else:
            if ui_elements.forward_checkbox_value.get() == 1:
                y_axis1_list.append(data["FWD:" + dictionary_signal_name])
            if ui_elements.reverse_checkbox_value.get() == 1:
                y_axis1_list.append(data["REV:" + dictionary_signal_name])

    if not y_axis1_list:
        y_axis1_list = y_axis2_list
    if len(axes) > 1 and not y_axis2_list:
        y_axis2_list = y_axis1_list

    scaling_factors = calculate_scaling_factors(x_axis, y_axis1_list, y_axis2_list)
    reset_figure_scale(axes, scaling_factors)


def create_popup_figure(
    ui_elements, figure, log_content, signal_name, home_scaling_factors
):
    child_w = tk.Toplevel(
        ui_elements.root, background=ui_elements.background_color, padx=10
    )
    child_w.geometry("750x450")
    child_w.title(figure.axes[0].get_title())

    f = create_figure(child_w, figure.axes[0].get_title(), ui_elements.background_color)
    f.toolbar.pack(side=tk.LEFT)
    f.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    f.toolbar.log_content = log_content
    f.toolbar.ui_elements = ui_elements
    f.toolbar.signal_name = signal_name
    f.toolbar.home_scaling_factors = home_scaling_factors
    update_figure(f, log_content)


def calculate_quench(log_content):
    result = {
        "energy": False,
        "bandwidth": False,
        "amplitude": False,
        "majority": False,
    }

    if (
        np.max(log_content.data["QUENCH:SYSTEM"]) - log_content.data["QUENCH_THRESH"]
        > 0
    ):
        result["energy"] = True

    return result


def create_classification_window(ui_elements, log_content):
    child_w = tk.Toplevel(
        ui_elements.root, background=ui_elements.background_color, padx=10, pady=10
    )
    child_w.title("Quench classification")
    tk.Entry(child_w, textvariable=tk.StringVar(value="Energy balance"), width=20).grid(
        row=0, column=1, sticky="NW"
    )
    tk.Entry(
        child_w, textvariable=tk.StringVar(value="Bandwidth estimation"), width=20
    ).grid(row=1, column=1, sticky="NW")
    tk.Entry(
        child_w, textvariable=tk.StringVar(value="Amplitude decay"), width=20
    ).grid(row=2, column=1, sticky="NW")
    tk.Entry(child_w, textvariable=tk.StringVar(value="Majority"), width=20).grid(
        row=3, column=1, sticky="NW"
    )
    radio_frame = tk.Frame(child_w, background=ui_elements.background_color)
    radio_frame.grid(row=4, column=1, columnspan=2, sticky="NW")

    result = calculate_quench(log_content)

    accept = tk.BooleanVar(value=result["majority"])
    ttk.Radiobutton(radio_frame, variable=accept, value=True, text="Quench").pack(
        side=tk.LEFT
    )
    ttk.Radiobutton(radio_frame, variable=accept, value=False, text="No quench").pack(
        side=tk.LEFT
    )
    child_w.rowconfigure(5, minsize=10)
    ttk.Button(
        child_w, text="Record", width=20, command=lambda: print(accept.get())
    ).grid(row=6, column=1, columnspan=2, sticky="NW")

    if result["energy"]:
        tk.Entry(
            child_w,
            textvariable=tk.StringVar(value="No Quench"),
            background="LightGreen",
        ).grid(row=0, column=2, sticky="NE")
    else:
        tk.Entry(
            child_w, textvariable=tk.StringVar(value="Quench"), background="tomato"
        ).grid(row=0, column=2, sticky="NE")

    if result["bandwidth"]:
        tk.Entry(
            child_w,
            textvariable=tk.StringVar(value="No Quench"),
            background="LightGreen",
        ).grid(row=1, column=2, sticky="NE")
    else:
        tk.Entry(
            child_w, textvariable=tk.StringVar(value="Quench"), background="tomato"
        ).grid(row=1, column=2, sticky="NE")
    if result["amplitude"]:
        tk.Entry(
            child_w,
            textvariable=tk.StringVar(value="No Quench"),
            background="LightGreen",
        ).grid(row=2, column=2, sticky="NE")
    else:
        tk.Entry(
            child_w, textvariable=tk.StringVar(value="Quench"), background="tomato"
        ).grid(row=2, column=2, sticky="NE")
    if result["majority"]:
        tk.Entry(
            child_w,
            textvariable=tk.StringVar(value="No Quench"),
            background="LightGreen",
        ).grid(row=3, column=2, sticky="NE")
    else:
        tk.Entry(
            child_w, textvariable=tk.StringVar(value="Quench"), background="tomato"
        ).grid(row=3, column=2, sticky="NE")


def _bound_to_mousewheel(event, canvas):
    canvas.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e, canvas))


def _unbound_to_mousewheel(event, canvas):
    canvas.unbind_all("<MouseWheel>")


def _on_mousewheel(event, canvas):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def create_parameter_display_window(ui_elements, log_content):
    child_w = tk.Toplevel(ui_elements.root, background=ui_elements.background_color)
    child_w.title("Parameters")
    container2 = tk.Frame(child_w, bg=ui_elements.background_color)
    container2.columnconfigure(0, weight=1)
    container2.rowconfigure(0, weight=1)
    container2.pack(fill="both", expand=True)
    canvas = tk.Canvas(
        container2,
        bg=ui_elements.background_color,
        highlightcolor=ui_elements.background_color,
        highlightbackground=ui_elements.background_color,
    )
    scrollbar = tk.Scrollbar(container2, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = tk.Frame(
        canvas, bg=ui_elements.background_color, pady=10, padx=10
    )
    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.bind("<Configure>", lambda event: frame_width(event, canvas, canvas_frame))
    scrollbar.grid(column=1, row=0, sticky="NS")
    canvas.grid(column=0, row=0, sticky="NSEW")
    scrollable_frame.bind("<Enter>", lambda e: _bound_to_mousewheel(e, canvas))
    scrollable_frame.bind("<Leave>", lambda e: _unbound_to_mousewheel(e, canvas))

    index = 0
    for key, value in log_content.data.items():
        if type(value) is float:
            index = index + 1
            b1 = tk.Entry(
                scrollable_frame, textvariable=tk.StringVar(value=key), width=40
            )
            b1.grid(row=index, column=1, sticky="NW")
            b2 = tk.Entry(scrollable_frame, textvariable=tk.StringVar(value=value))  # type: ignore
            b2.grid(row=index, column=2, sticky="NE")


def update_window(log_content, ui_elements):
    update_figures(ui_elements.figures, log_content)
    ui_elements.title_sv.set(log_content.log_title)
    ui_elements.date_sv.set(log_content.data.get("ACQ_FLT_TS"))
    ui_elements.ticks_sv.set(f"{len(log_content.data['CAV:FLTTWF'])} ticks")
    ui_elements.path_text_sv.set(log_content.filename)
    ui_elements.root.wm_title(log_content.filename)
    ui_elements.parameters_button.configure(
        command=lambda: create_parameter_display_window(ui_elements, log_content)
    )
    ui_elements.classification_button.configure(
        command=lambda: create_classification_window(ui_elements, log_content)
    )


def update_figure(figure, log_content):
    data = log_content.data

    x_axis = data["CAV:FLTTWF"]

    if figure.ax.get_title() == "Amplitude":
        figure.line_cav.set_data(x_axis, data["CAV:FLTAWF"])
        figure.line_fwd.set_data(x_axis, data["FWD:FLTAWF"])
        figure.line_rev.set_data(x_axis, data["REV:FLTAWF"])
        figure.line_decay_ref.set_data(x_axis, data.get("DECAYREFWF"))
        if (
            data.get("CAV:FLTAWF.LOPR") is not None
            and data.get("CAV:FLTAWF.HOPR") is not None
            and data.get("FWD:FLTAWF.LOPR") is not None
            and data.get("FWD:FLTAWF.HOPR") is not None
        ):
            scaling = calculate_scaling_factors(
                x_axis,
                [data["CAV:FLTAWF.LOPR"], data["CAV:FLTAWF.HOPR"]],
                [data["FWD:FLTAWF.LOPR"], data["FWD:FLTAWF.HOPR"]],
            )
        else:
            scaling = calculate_scaling_factors(
                x_axis, data["CAV:FLTAWF"], [data["FWD:FLTAWF"], data["REV:FLTAWF"]]
            )
        figure.toolbar.home_scaling_factors = scaling
        reset_figure_scale([figure.ax, figure.ax2], scaling)
        figure.canvas.draw()
    elif figure.ax.get_title() == "Phase":
        figure.line_cav.set_data(x_axis, data["CAV:FLTPWF"])
        figure.line_fwd.set_data(x_axis, data["FWD:FLTPWF"])
        figure.line_rev.set_data(x_axis, data["REV:FLTPWF"])
        figure.toolbar.home_scaling_factors = calculate_scaling_factors(
            x_axis, [-180, 180]
        )
        reset_figure_scale([figure.ax], figure.toolbar.home_scaling_factors)
        figure.ax.set_yticks([-180, 0, 180])
        figure.canvas.draw()
    elif figure.ax.get_title() == "Power":
        figure.line_cav.set_data(x_axis, data["CAV:FLTPWRWF"])
        figure.line_fwd.set_data(x_axis, data["FWD:FLTPWRWF"])
        figure.line_rev.set_data(x_axis, data["REV:FLTPWRWF"])
        if (
            data.get("FWD:FLTPWRWF.LOPR") is not None
            and data.get("FWD:FLTPWRWF.HOPR") is not None
        ):
            scaling = calculate_scaling_factors(
                x_axis,
                [data["CAV:FLTPWRWF"]],
                [data["FWD:FLTPWRWF.LOPR"], data["FWD:FLTPWRWF.HOPR"]],
            )
        else:
            scaling = calculate_scaling_factors(
                x_axis,
                [data["CAV:FLTPWRWF"]],
                [data["FWD:FLTPWRWF"], data["REV:FLTPWRWF"]],
            )
        figure.toolbar.home_scaling_factors = scaling
        reset_figure_scale([figure.ax, figure.ax2], figure.toolbar.home_scaling_factors)
        figure.canvas.draw()
    elif figure.ax.get_title() == "I":
        figure.line_cav.set_data(x_axis, data["CAV:FLTIWF"])
        figure.line_fwd.set_data(x_axis, data["FWD:FLTIWF"])
        figure.line_rev.set_data(x_axis, data["REV:FLTIWF"])

        figure.line_l.set_data(x_axis, data.get("CTRL:FLTLIMS_IL"))
        figure.line_h.set_data(x_axis, data.get("CTRL:FLTLIMS_IH"))
        if (
            data.get("CTRL:FLTLIMS_IL") is not None
            and data.get("CTRL:FLTLIMS_IH") is not None
        ):
            figure.toolbar.home_scaling_factors = calculate_scaling_factors(
                x_axis,
                [
                    data["CAV:FLTIWF"],
                    data["FWD:FLTIWF"],
                    data["REV:FLTIWF"],
                    data["CTRL:FLTLIMS_IL"],
                    data["CTRL:FLTLIMS_IH"],
                ],
            )
        else:
            figure.toolbar.home_scaling_factors = calculate_scaling_factors(
                x_axis, [data["CAV:FLTIWF"], data["FWD:FLTIWF"], data["REV:FLTIWF"]]
            )
        reset_figure_scale([figure.ax], figure.toolbar.home_scaling_factors)
        figure.canvas.draw()
    elif figure.ax.get_title() == "Q":
        figure.line_cav.set_data(x_axis, data["CAV:FLTQWF"])
        figure.line_fwd.set_data(x_axis, data["FWD:FLTQWF"])
        figure.line_rev.set_data(x_axis, data["REV:FLTQWF"])
        figure.line_l.set_data(x_axis, data.get("CTRL:FLTLIMS_QL"))
        figure.line_h.set_data(x_axis, data.get("CTRL:FLTLIMS_QH"))
        if (
            data.get("CTRL:FLTLIMS_QL") is not None
            and data.get("CTRL:FLTLIMS_QH") is not None
        ):
            figure.toolbar.home_scaling_factors = calculate_scaling_factors(
                x_axis,
                [
                    data["CAV:FLTQWF"],
                    data["FWD:FLTQWF"],
                    data["REV:FLTQWF"],
                    data["CTRL:FLTLIMS_QL"],
                    data["CTRL:FLTLIMS_QH"],
                ],
            )
        else:
            figure.toolbar.home_scaling_factors = calculate_scaling_factors(
                x_axis, [data["CAV:FLTQWF"], data["FWD:FLTQWF"], data["REV:FLTQWF"]]
            )
        reset_figure_scale([figure.ax], figure.toolbar.home_scaling_factors)
    elif figure.ax.get_title() == "Quench detect":
        figure.line_cavity_du.set_data(data["QUENCH:FLTTWF"], data["QUENCH:CAVITY"])
        figure.line_system.set_data(data["QUENCH:FLTTWF"], data["QUENCH:SYSTEM"])
        figure.line_waveguide.set_data(data["QUENCH:FLTTWF"], data["QUENCH:WAVEGUIDE"])
        figure.line_threshold.set_data(
            data["QUENCH:FLTTWF"], data.get("QUENCH:THRESHOLD")
        )
        figure.toolbar.home_scaling_factors = calculate_scaling_factors(
            data["QUENCH:FLTTWF"],
            [data["QUENCH:CAVITY"], data["QUENCH:SYSTEM"], data["QUENCH:WAVEGUIDE"]],
        )
        reset_figure_scale([figure.ax], figure.toolbar.home_scaling_factors)

    figure.canvas.draw()


def update_figures(figures, log_content):
    for figure in figures.values():
        figure.toolbar.log_content = log_content
        update_figure(figure, log_content)


def open_file_action(ui_elements):
    filename = askopenfilename()
    log_content = parse_file(filename)
    if log_content is not None:
        update_window(log_content, ui_elements)
    print(filename)


def listbox_onselect(listbox, directory, ui_elements, radio_value):
    try:
        index = int(listbox.curselection()[0])
    except IndexError:
        return
    value = listbox.get(index)
    if radio_value.get() == 1:
        actual_directory = join(directory, value)
        filename = join(actual_directory, value + ".txt")
        if not isfile(filename):
            showwarning(
                title="File not found", message=f"The file {filename} does not exist."
            )
            return
    else:
        filename = join(directory, value)
    log_content = parse_file(filename)
    if log_content is not None:
        update_window(log_content, ui_elements)
    print(filename)


def modify_entry_action(
    sv, onlyfiles, onlydirectories, listbox, case_checkbox_value, radio_value
):
    if radio_value.get() == 1:
        files = onlydirectories
    else:
        files = onlyfiles

    if case_checkbox_value.get() == 1:
        filtered_files = [k for k in files if sv.get() in k]
    else:
        lower_files = [file.lower() for file in files]
        filtered_files = []
        for idx, file in enumerate(lower_files):
            if sv.get().lower() in file:
                filtered_files.append(files[idx])
    listbox.delete(0, tk.END)
    filtered_files.sort(reverse=True)
    for index, element in enumerate(filtered_files):
        listbox.insert(index, element)


def enter_event_action(listbox, directory, ui_elements, radio_value):
    listbox.selection_clear(0, listbox.size())
    elements = list(listbox.get(0, listbox.size()))
    listbox.selection_set(elements.index(listbox.get(tk.ACTIVE)))
    listbox_onselect(listbox, directory, ui_elements, radio_value)


def open_directory_window(dir_path, ui_elements):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    onlydirectories = [
        f
        for f in listdir(dir_path)
        if isdir(join(dir_path, f)) and isfile(join(dir_path, f, f + ".txt"))
    ]
    elements_list = onlydirectories
    child_w = tk.Toplevel(ui_elements.root)
    child_w.geometry("750x250")
    child_w.title("Select file")
    filter_frame = ttk.Frame(child_w)
    filter_frame.pack(side=tk.TOP, expand=False, fill=tk.X, padx=10, pady=10)
    label = ttk.Label(filter_frame, text="Filter")
    label.pack(side=tk.LEFT)
    sv = tk.StringVar()
    entry = ttk.Entry(filter_frame, textvariable=sv)
    entry.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    case_checkbox_value = tk.IntVar()
    radio_value = tk.IntVar()

    case_checkbox_value.set(0)
    case_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_entry_action(
            sv, onlyfiles, onlydirectories, listbox, case_checkbox_value, radio_value
        ),
    )
    case_checkbox = ttk.Checkbutton(
        filter_frame, text="Case sensitive", variable=case_checkbox_value
    )
    case_checkbox.pack(side=tk.LEFT)

    radio_frame = tk.Frame(child_w)
    radio_frame.pack(anchor=tk.W)
    option_dirs_radio = ttk.Radiobutton(
        radio_frame,
        text="Show directories",
        variable=radio_value,
        value=1,
        command=lambda: modify_entry_action(
            sv, onlyfiles, onlydirectories, listbox, case_checkbox_value, radio_value
        ),
    )
    option_dirs_radio.pack(anchor=tk.W)
    option_files_radio = ttk.Radiobutton(
        radio_frame,
        text="Show files",
        variable=radio_value,
        value=2,
        command=lambda: modify_entry_action(
            sv, onlyfiles, onlydirectories, listbox, case_checkbox_value, radio_value
        ),
    )
    option_files_radio.pack(anchor=tk.W)
    radio_value.set(1)

    listbox_frame = tk.Frame(child_w)
    listbox_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    listbox = tk.Listbox(listbox_frame, width=40, height=10, selectmode=tk.SINGLE)
    listbox.bind(
        "<<ListboxSelect>>",
        lambda event: listbox_onselect(listbox, dir_path, ui_elements, radio_value),
    )
    sv.trace(
        "w",
        callback=lambda name, index, mode: modify_entry_action(
            sv, onlyfiles, onlydirectories, listbox, case_checkbox_value, radio_value
        ),
    )
    child_w.bind(
        "<Return>",
        lambda event: enter_event_action(listbox, dir_path, ui_elements, radio_value),
    )
    elements_list.sort(reverse=True)
    for index, element in enumerate(elements_list):
        listbox.insert(index, element)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
    scrollbar.config(command=listbox.yview)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    child_w.after(1000, lambda: child_w.lift())


def open_directory_action(ui_elements):
    dir_path = askdirectory(mustexist=True)
    if dir_path:
        open_directory_window(dir_path, ui_elements)


def create_figure(master, title, background_color=None):
    """
    Create a figure object with settings that depend on the title

    :param master: tkinter master
    :param str title: figure title
    :param background_color: color string for the canvas background
    :type background_color: str or None
    :returns: FigureObject containing all things related to a figure
    :rtype: FigureObject
    """

    x_data = []
    y_data = []

    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.title.set_size(10)  # type: ignore
    ax.ticklabel_format(axis="x", style="sci", scilimits=[-2, 3])  # type: ignore
    ax2 = None
    line_l = None
    line_h = None
    line_decay_ref = None
    line_cavity_du = None
    line_system = None
    line_waveguide = None
    line_threshold = None
    line_cav, line_fwd, line_rev = (None, None, None)

    color1 = "#0000e0"
    color2 = "#a00000"
    color3 = "#C08448"
    color4 = "#00C0C0"
    color5 = "#9EC4E0"

    if title == "Amplitude":
        ax.set_ylabel("MV")
        ax2 = ax.twinx()
        (line_cav,) = ax.plot(x_data, y_data, linewidth=1, color=color1)
        (line_fwd,) = ax2.plot(x_data, y_data, linewidth=1, color=color2)
        (line_rev,) = ax2.plot(x_data, y_data, linewidth=1, color=color3)
        (line_decay_ref,) = ax.plot(x_data, y_data, linewidth=1, color=color5)
        ax2.set_ylabel("sqrt(W)")
    elif title == "Phase":
        ax.set_ylabel("degrees")
        (line_cav,) = ax.plot(x_data, y_data, linewidth=1, color=color1)
        (line_fwd,) = ax.plot(x_data, y_data, linewidth=1, color=color2)
        (line_rev,) = ax.plot(x_data, y_data, linewidth=1, color=color3)
    elif title == "Power":
        ax.set_ylabel("mW")
        ax2 = ax.twinx()
        (line_cav,) = ax.plot(x_data, y_data, linewidth=1, color=color1)
        (line_fwd,) = ax2.plot(x_data, y_data, linewidth=1, color=color2)
        (line_rev,) = ax2.plot(x_data, y_data, linewidth=1, color=color3)
        ax2.set_ylabel("W")
    elif title in ("I", "Q"):
        (line_cav,) = ax.plot(x_data, y_data, linewidth=1, color=color1)
        (line_fwd,) = ax.plot(x_data, y_data, linewidth=1, color=color2)
        (line_rev,) = ax.plot(x_data, y_data, linewidth=1, color=color3)
        (line_l,) = ax.plot(x_data, y_data, linewidth=1, color=color4)
        (line_h,) = ax.plot(x_data, y_data, linewidth=1, color=color4)
    elif title == "Quench detect":
        (line_cavity_du,) = ax.plot(x_data, y_data, linewidth=1)
        (line_system,) = ax.plot(x_data, y_data, linewidth=1)
        (line_waveguide,) = ax.plot(x_data, y_data, linewidth=1)
        (line_threshold,) = ax.plot(x_data, y_data, linewidth=1)
    else:
        (line_cav,) = ax.plot(x_data, y_data, linewidth=1, color=color1)
        (line_fwd,) = ax.plot(x_data, y_data, linewidth=1, color=color2)
        (line_rev,) = ax.plot(x_data, y_data, linewidth=1, color=color3)

    ax.set_xlabel("seconds")

    fig.tight_layout()
    if background_color is not None:
        fig.patch.set_facecolor(background_color)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar(
        canvas,
        master,
        pack_toolbar=False,
        vertical_toolbar=True,
        background_color=background_color,
    )
    # toolbar.remove_rubberband()
    toolbar.update()
    canvas.mpl_connect("key_press_event", key_press_handler)  # type: ignore

    return FigureObject(
        line_cav,
        line_fwd,
        line_rev,
        canvas,
        toolbar,
        ax,
        ax2,
        line_l,
        line_h,
        line_decay_ref,
        line_cavity_du,
        line_system,
        line_waveguide,
        line_threshold,
    )


def modify_lines_action(figures, checkbox_value, line_string):
    if checkbox_value.get() == 1:
        for figure in figures.values():
            line = getattr(figure, line_string)
            if line is not None:
                line.set_visible(True)
                figure.canvas.draw()
    else:
        for figure in figures.values():
            line = getattr(figure, line_string)
            if line is not None:
                line.set_visible(False)
                figure.canvas.draw()


def frame_width(event, canvas, canvas_frame):
    canvas_width = event.width
    canvas.itemconfig(canvas_frame, width=canvas_width)


def menu_popup(event, copy_menu):
    try:
        copy_menu.tk_popup(event.x_root, event.y_root)
    finally:
        copy_menu.grab_release()


def create_main_window(file_path=None, dir_path=None):
    background_color = "LightGray"

    root = tk.Tk()
    root.config(bg=background_color)
    root.wm_title("Fault Visualization Tool")
    root.columnconfigure(0, weight=1, minsize=300)
    root.rowconfigure(1, weight=1)

    container_frame = tk.Frame(root, bg=background_color)
    container_frame.columnconfigure(0, weight=1)
    container_frame.rowconfigure(1, weight=1)
    container_frame.grid(column=0, row=1, sticky="NSEW")

    screen_height = root.winfo_screenheight()
    window_height = 900
    if screen_height < window_height:
        window_height = screen_height - screen_height * 0.05

    frame = tk.Frame(container_frame, bg=background_color, height=window_height)
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(2, weight=1)

    canvas = tk.Canvas(
        container_frame,
        bg=background_color,
        highlightcolor=background_color,
        highlightbackground=background_color,
    )
    scrollbar = tk.Scrollbar(container_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = tk.Frame(canvas, bg=background_color)
    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.bind("<Configure>", lambda event: frame_width(event, canvas, canvas_frame))
    scrollbar.grid(column=1, row=1, sticky="NS")
    canvas.grid(column=0, row=1, sticky="NSEW")

    scrollable_frame.columnconfigure(1, minsize=10)
    scrollable_frame.columnconfigure(0, minsize=500, weight=1)
    scrollable_frame.columnconfigure(2, minsize=500, weight=1)

    top_frame = tk.Frame(root, bg=background_color)
    buttons_frame = tk.Frame(top_frame, bg=background_color)
    checkboxes_frame1 = tk.Frame(top_frame, bg=background_color)
    checkboxes_frame2 = tk.Frame(top_frame, bg=background_color)

    title_sv = tk.StringVar(top_frame, value="")
    title = tk.Label(top_frame, textvariable=title_sv, width=50)
    date_sv = tk.StringVar(top_frame, value="")
    date = tk.Label(top_frame, textvariable=date_sv, width=20)
    ticks_sv = tk.StringVar(top_frame, value="")
    # ticks = tk.Label(top_frame, textvariable=ticks_sv, width=20)

    path_text_sv = tk.StringVar()
    path_text_sv.set("")
    path_text = tk.Entry(
        top_frame,
        textvariable=path_text_sv,
        state="readonly",
        width=10,
        justify="center",
        relief="flat",
    )
    path_text.grid(column=6, row=1, sticky="NEW", columnspan=3)
    copy_menu = tk.Menu(root, tearoff=0)
    if pyperclip_imported:
        copy_menu.add_command(
            label="Copy path", command=lambda: pyperclip.copy(path_text_sv.get())
        )
        path_text.bind("<Button-3>", lambda event: menu_popup(event, copy_menu))

    buttons_frame.grid(column=0, row=0, sticky="NW")
    top_frame.grid(column=0, row=0, sticky="NW", padx=10, pady=10)
    top_frame.columnconfigure(1, minsize=100)
    checkboxes_frame1.grid(column=2, row=0, sticky="NW", rowspan=2)
    top_frame.columnconfigure(3, minsize=10)
    checkboxes_frame2.grid(column=4, row=0, sticky="NW", rowspan=2)
    top_frame.columnconfigure(5, minsize=100)
    title.grid(column=6, row=0, sticky="N")
    top_frame.columnconfigure(7, minsize=100)
    date.grid(column=8, row=0, sticky="N")
    # ticks.grid(column=6, row=1, sticky="N")

    figure_amplitude_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)
    figure_phase_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)
    figure_power_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)
    figure_i_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)
    figure_q_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)
    figure_quench_frame = tk.Frame(scrollable_frame, bg=background_color, padx=10)

    # scrollable_frame.pack(fill=tk.BOTH)

    figure_amplitude = create_figure(
        figure_amplitude_frame, "Amplitude", background_color
    )
    figure_phase = create_figure(figure_phase_frame, "Phase", background_color)
    figure_power = create_figure(figure_power_frame, "Power", background_color)
    figure_i = create_figure(figure_i_frame, "I", background_color)
    figure_q = create_figure(figure_q_frame, "Q", background_color)
    figure_quench = create_figure(
        figure_quench_frame, "Quench detect", background_color
    )

    figures = {
        "amplitude": figure_amplitude,
        "phase": figure_phase,
        "power": figure_power,
        "i": figure_i,
        "q": figure_q,
        "quench": figure_quench,
    }

    cavity_checkbox_value = tk.IntVar(value=1)
    forward_checkbox_value = tk.IntVar(value=1)
    reverse_checkbox_value = tk.IntVar(value=1)
    decay_checkbox_value = tk.IntVar(value=1)
    cavity_du_checkbox_value = tk.IntVar(value=1)
    system_checkbox_value = tk.IntVar(value=1)
    waveguide_checkbox_value = tk.IntVar(value=1)

    ttk.Style().configure(
        "CavityCheckbox.TCheckbutton",
        foreground=figure_amplitude.line_cav.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "ForwardCheckbox.TCheckbutton",
        foreground=figure_amplitude.line_fwd.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "ReverseCheckbox.TCheckbutton",
        foreground=figure_amplitude.line_rev.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "DecayCheckbox.TCheckbutton",
        foreground=figure_amplitude.line_decay_ref.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "CavityDUCheckbox.TCheckbutton",
        foreground=figure_quench.line_cavity_du.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "SystemCheckbox.TCheckbutton",
        foreground=figure_quench.line_system.get_color(),  # type: ignore
        background=background_color,
    )
    ttk.Style().configure(
        "WaveguideCheckbox.TCheckbutton",
        foreground=figure_quench.line_waveguide.get_color(),  # type: ignore
        background=background_color,
    )

    cavity_checkbox = ttk.Checkbutton(
        checkboxes_frame1,
        text="Cavity",
        variable=cavity_checkbox_value,
        style="CavityCheckbox.TCheckbutton",
    )
    forward_checkbox = ttk.Checkbutton(
        checkboxes_frame1,
        text="Forward",
        variable=forward_checkbox_value,
        style="ForwardCheckbox.TCheckbutton",
    )
    reverse_checkbox = ttk.Checkbutton(
        checkboxes_frame1,
        text="Reverse",
        variable=reverse_checkbox_value,
        style="ReverseCheckbox.TCheckbutton",
    )
    decay_checkbox = ttk.Checkbutton(
        checkboxes_frame1,
        text="Decay",
        variable=decay_checkbox_value,
        style="DecayCheckbox.TCheckbutton",
    )
    cavity_du_checkbox = ttk.Checkbutton(
        checkboxes_frame2,
        text="Cavity -dU/dt",
        variable=cavity_du_checkbox_value,
        style="CavityDUCheckbox.TCheckbutton",
    )
    system_checkbox = ttk.Checkbutton(
        checkboxes_frame2,
        text="System net",
        variable=system_checkbox_value,
        style="SystemCheckbox.TCheckbutton",
    )
    waveguide_checkbox = ttk.Checkbutton(
        checkboxes_frame2,
        text="Waveguide net",
        variable=waveguide_checkbox_value,
        style="WaveguideCheckbox.TCheckbutton",
    )

    cavity_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, cavity_checkbox_value, "line_cav"
        ),
    )
    forward_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, forward_checkbox_value, "line_fwd"
        ),
    )
    reverse_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, reverse_checkbox_value, "line_rev"
        ),
    )
    decay_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, decay_checkbox_value, "line_decay_ref"
        ),
    )
    cavity_du_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, cavity_du_checkbox_value, "line_cavity_du"
        ),
    )
    system_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, system_checkbox_value, "line_system"
        ),
    )
    waveguide_checkbox_value.trace(
        "w",
        lambda name, index, mode: modify_lines_action(
            figures, waveguide_checkbox_value, "line_waveguide"
        ),
    )

    cavity_checkbox.pack(anchor="nw")
    forward_checkbox.pack(anchor="nw")
    reverse_checkbox.pack(anchor="nw")
    decay_checkbox.pack(anchor="nw")
    cavity_du_checkbox.pack(anchor="nw")
    system_checkbox.pack(anchor="nw")
    waveguide_checkbox.pack(anchor="nw")

    ui_elements = UIElements(
        root,
        title_sv,
        date_sv,
        ticks_sv,
        figures,
        path_text_sv,
        cavity_checkbox_value,
        forward_checkbox_value,
        reverse_checkbox_value,
        decay_checkbox_value,
        cavity_du_checkbox_value,
        system_checkbox_value,
        waveguide_checkbox_value,
        background_color,
    )

    for key, value in figures.items():
        value.toolbar.ui_elements = ui_elements  # type: ignore
        value.toolbar.signal_name = key  # type: ignore

    ttk.Style().configure("bg.TButton", background=background_color)
    row1_buttons_frame = tk.Frame(buttons_frame, bg=background_color)
    row1_buttons_frame.pack(anchor="nw")
    row2_buttons_frame = tk.Frame(buttons_frame, bg=background_color)
    row2_buttons_frame.pack(anchor="nw")
    ttk.Button(
        row1_buttons_frame, text="Quit", command=root.quit, style="bg.TButton"
    ).pack(side=tk.LEFT, anchor="nw")
    ttk.Button(
        row1_buttons_frame,
        text="Open file",
        command=lambda: open_file_action(ui_elements),
        style="bg.TButton",
    ).pack(side=tk.LEFT, anchor="nw")
    ttk.Button(
        row1_buttons_frame,
        text="Open directory",
        command=lambda: open_directory_action(ui_elements),
        style="bg.TButton",
    ).pack(side=tk.LEFT, anchor="nw")
    parameters_button = ttk.Button(
        row2_buttons_frame,
        text="View parameters",
        command=lambda: showwarning(
            title="File not found", message="You need to open a file first."
        ),
        style="bg.TButton",
    )
    parameters_button.pack(side=tk.LEFT, anchor="nw")
    ui_elements.parameters_button = parameters_button
    classification_button = ttk.Button(
        row2_buttons_frame,
        text="Quench classification",
        command=lambda: showwarning(
            title="File not found", message="You need to open a file first."
        ),
        style="bg.TButton",
    )
    classification_button.pack(side=tk.LEFT, anchor="nw")
    ui_elements.classification_button = classification_button

    frame.grid(column=0, row=1, padx=10, pady=10, sticky="NSEW")

    figure_amplitude_frame.grid(column=0, row=1, sticky="NSEW")
    figure_amplitude.toolbar.pack(side=tk.LEFT)
    figure_amplitude.canvas.get_tk_widget().pack(
        side=tk.LEFT, fill=tk.BOTH, expand=True
    )

    figure_phase_frame.grid(column=0, row=2, sticky="NSEW")
    figure_phase.toolbar.pack(side=tk.LEFT)
    figure_phase.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    figure_power_frame.grid(column=0, row=3, sticky="NSEW")
    figure_power.toolbar.pack(side=tk.LEFT)
    figure_power.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    figure_i_frame.grid(column=2, row=1, sticky="NSEW")
    figure_i.toolbar.pack(side=tk.LEFT)
    figure_i.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    figure_q_frame.grid(column=2, row=2, sticky="NSEW")
    figure_q.toolbar.pack(side=tk.LEFT)
    figure_q.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    figure_quench_frame.grid(column=2, row=3, sticky="NSEW")
    figure_quench.toolbar.pack(side=tk.LEFT)
    figure_quench.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    if file_path is not None:
        log_content = parse_file(file_path)
        if log_content is not None:
            update_window(log_content, ui_elements)

    if dir_path is not None:
        open_directory_window(dir_path, ui_elements)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fault Visualization Tool.",
        epilog="Both arguments can be passed at the same time.",
    )
    parser.add_argument(
        "-d", "--dir", help="open a selection window at the path specified"
    )
    parser.add_argument("-f", "--file", help="open the file specified")
    args = parser.parse_args()

    file_path = args.file
    if file_path is not None:
        file = Path(file_path)
        if not file.is_file():
            file_path = None
            showwarning(
                title="File not found",
                message="The file passed via command line does not exist. Use the interface or retry.",
            )

    dir_path = args.dir
    if dir_path is not None:
        directory = Path(dir_path)
        if not directory.is_dir():
            dir_path = None
            showwarning(
                title="Directory not found",
                message="The directory passed via command line does not exist. Use the interface or retry.",
            )

    create_main_window(file_path, dir_path)
