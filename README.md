# holography_python
calculate and simulate holograms in python<br>
the example shows the propagation of the plane image shepp logan phantom
<img src="shepp_logan_phantom.png">
to a complex wavefield "wavefield.npy".<br>
This wavefield is stored for display in a 4F and RELPH setup.
<img src="4F.png">
<img src="RELPH.png">

The 4F setup modulates the phase with two phase modulators, where the first one modulates the phase $\phi$ and the second one modulates the amplitude by interference with itself and correcting the phase error with the first modulator as follows:
<img src="https://latex.codecogs.com/gif.latex?\phi2 = \pm 2 \operatorname{arccos}(ampli)" />
(amplitude modulated by interference of light with itself)
<img src="https://latex.codecogs.com/gif.latex?\phi1 = \operatorname{phase}(E) - \phi2" />
(phase + phase correction for this particular amplitude modulation)
where the wavefield is reproduced according to following equation
<img src="https://latex.codecogs.com/gif.latex?E = \exp(i \phi1) \underbrace{(0.5 + 0.5*\exp(-i \phi2))}_{\text{amplitude modulation}}" />

The RELPH setup modulates the wavefield using two identical phase modulators with a phase $\phi_{1/2}$
<img src="https://latex.codecogs.com/gif.latex?\phi_{1/2} = \operatorname{phase}(E) \pm \operatorname{atan}\left(\sqrt{4/|E^2|-1}\right)" />
<img src="https://latex.codecogs.com/gif.latex?E = \exp(i \phi1) + \exp(i \phi2)" />

phase 1 is encoded with a red and phase 2 with a green color.
