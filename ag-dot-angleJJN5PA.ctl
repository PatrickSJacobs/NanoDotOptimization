(define-param sy 250.038) ; size of cell in Y direction 
(define-param sz sy) ; size of cell in z direction 
(define-param sx 6) ; size of cell in X direction

(define-param sl (* sy 0.7));  source location from the pml

(define-param hx 0.1) ; size of open on metal or dielectric in X direction                              
(define-param hy sy) ; size of open on metal or dielectric in Y direction 
(define-param hz sz) ; size of open on metal or dielectric in z direction 

(define-param efield Ez)
(define-param dpml 0.4) ; Thickness of PML

(define-param sr 0.019)
(define-param ht 0.088)

(define-param theta_deg 0)     ; angle in degrees.


(set-param! eps-averaging? false)

(define Si
	(make dielectric (epsilon 12)))
(define a-si
      (make dielectric (epsilon 1.2518)
            (polarizations
             (make polarizability
               (omega -2.1762) (gamma 2.3364) (sigma -10.4548))

	     (make polarizability
               (omega 3.0452) (gamma 2.0402) (sigma 22.332))
)))

;----------------------------------------

 (define silver_f
          (make dielectric (epsilon 1)
                  (E-susceptibilities
                     (make lorentzian-susceptibility
                       (frequency 1.000e-20) (gamma 0.00615) (sigma 4.444e+41  ))
                     (make lorentzian-susceptibility
                       (frequency 0.10453) (gamma 0.49782) (sigma 7.92470  ))
                     (make lorentzian-susceptibility
                       (frequency 0.57404) (gamma 0.05790) (sigma 0.50133  ))
                     (make lorentzian-susceptibility
                       (frequency 1.04854) (gamma 0.00833) (sigma 0.01333  ))
                     (make lorentzian-susceptibility
                       (frequency 1.16358) (gamma 0.11734) (sigma 0.82655  ))
                     (make lorentzian-susceptibility
                       (frequency 2.59926) (gamma 0.30989) (sigma 1.11334  ))
                  )
          )
 )
;----------------------------------------

; silver (Ag) from meep-material fold

(define Ag-plasma-frq (* 9.01 eV-um-scale))

(define Ag-f0 0.845)
(define Ag-frq0 1e-10)
(define Ag-gam0 (* 0.048 eV-um-scale))
(define Ag-sig0 (/ (* Ag-f0 (sqr Ag-plasma-frq)) (sqr Ag-frq0)))

(define Ag-f1 0.065)
(define Ag-frq1 (* 0.816 eV-um-scale)) ; 1.519 um
(define Ag-gam1 (* 3.886 eV-um-scale))
(define Ag-sig1 (/ (* Ag-f1 (sqr Ag-plasma-frq)) (sqr Ag-frq1)))

(define Ag-f2 0.124)
(define Ag-frq2 (* 4.481 eV-um-scale)) ; 0.273 um
(define Ag-gam2 (* 0.452 eV-um-scale))
(define Ag-sig2 (/ (* Ag-f2 (sqr Ag-plasma-frq)) (sqr Ag-frq2)))

(define Ag-f3 0.011)
(define Ag-frq3 (* 8.185 eV-um-scale)) ; 0.152 um
(define Ag-gam3 (* 0.065 eV-um-scale))
(define Ag-sig3 (/ (* Ag-f3 (sqr Ag-plasma-frq)) (sqr Ag-frq3)))

(define Ag-f4 0.840)
(define Ag-frq4 (* 9.083 eV-um-scale)) ; 0.137 um
(define Ag-gam4 (* 0.916 eV-um-scale))
(define Ag-sig4 (/ (* Ag-f4 (sqr Ag-plasma-frq)) (sqr Ag-frq4)))

(define Ag-f5 5.646)
(define Ag-frq5 (* 20.29 eV-um-scale)) ; 0.061 um
(define Ag-gam5 (* 2.419 eV-um-scale))
(define Ag-sig5 (/ (* Ag-f5 (sqr Ag-plasma-frq)) (sqr Ag-frq5)))

(define Ag (make medium (epsilon 1.0)
  (E-susceptibilities
     (make drude-susceptibility
       (frequency Ag-frq0) (gamma Ag-gam0) (sigma Ag-sig0))
     (make lorentzian-susceptibility
       (frequency Ag-frq1) (gamma Ag-gam1) (sigma Ag-sig1))
     (make lorentzian-susceptibility
       (frequency Ag-frq2) (gamma Ag-gam2) (sigma Ag-sig2))
     (make lorentzian-susceptibility
       (frequency Ag-frq3) (gamma Ag-gam3) (sigma Ag-sig3))
     (make lorentzian-susceptibility
       (frequency Ag-frq4) (gamma Ag-gam4) (sigma Ag-sig4))
     (make lorentzian-susceptibility
       (frequency Ag-frq5) (gamma Ag-gam5) (sigma Ag-sig5)))))

;------------------------------------------------------------------

(set! geometry-lattice (make lattice (size sx sy sz)))


(define-param no-metal? false) ; if true, have metal
(set! k-point (vector3 0 0 0))
(set! geometry
      (if no-metal?
          (list
           (make block
		(center 0 0 0)	      
		(size sx sy sz)
		(material air)
		))

	   (list
            
	     (make cylinder
		(center 0 0 0) 
		(height ht)
		(radius sr)
		(axis 1 0 0)
		(material Ag)
		)
)))

;(define-param fcen 1.67) ; pulse center frequency                               
;(define-param df 2)    ; pulse width (in frequency)                             

(define-param wvl-min 0.4)          ; minimum wavelength of source
(define-param wvl-max 0.8)          ; maximum wavelength of source
(define fmin (/ wvl-max))           ; minimum frequency of source
(define fmax (/ wvl-min))           ; maximum frequency of source
(define fcen (* 0.5 (+ fmin fmax))) ; center frequency of source
(define df (- fmax fmin))           ; frequency width of source

; pw-amp is a function that returns the amplitude exp(ik(x+x0)) at a
; given point x.  (We need the x0 because current amplitude functions
; in Meep are defined relative to the center of the current source,
; whereas we want a fixed origin.)  Actually, it is a function of k
; and x0 that returns a function of x ...

(define (pw-amp k x0) (lambda (x)
  (exp (* 0+1i (vector3-dot k (vector3+ x x0))))))

(define theta_rad (/ (* pi theta_deg) 180))

; direction of k (length is irrelevant)
(define-param kdir (vector3 (cos theta_rad) (sin theta_rad) 0))

; k with correct length
(define k (vector3-scale (* 2 pi fcen) (unit-vector3 kdir)))


(set! sources (list
       (make source
         (src (make gaussian-src (frequency fcen) (fwidth df)))
         (component efield)  (center (+ (+ dpml sl) (* -0.5 sx)) 0 0) (size 0 sy sz)
         (amp-func (pw-amp k (vector3  (+ (* -0.5 sx) dpml) 0 0)))
)))


(define-param resolu 100)
(set-param! resolution resolu)

(set! pml-layers (list (make pml (direction X) (thickness dpml))))

(define-param nfreq 400) ; number of frequencies at which to compute flux             
(define trans ; transmitted flux                                                
      (add-flux fcen df nfreq
                (if no-metal?
		  
                    (make flux-region
                     (center (- (/ sx 2) (+ dpml 0.1)) 0 0) (size 0 sy sz) )
                   
		    (make flux-region
                     (center (- (/ sx 2) (+ dpml 0.1)) 0 0) (size 0 sy sz)))))


(define refl ; reflected flux                                                   
      (add-flux fcen df nfreq
                 (make flux-region 
                   (center (+ (/ sx -2) (+ dpml sy)) 0 0) (size 0 sy sz))))


(if (not no-metal?) (load-minus-flux "refl-flux" refl))

(run-sources+ 500
 
(stop-when-fields-decayed 50 Ez
                               (vector3 (- (/ sx 2) (+ dpml 0.1)) 0 0)  1e-3)

(at-beginning (in-volume (volume (center 0 0) (size (- sx (* (+ dpml 0.1) 2)) sy sz )) output-epsilon))
(at-end (in-volume (volume (center 0 0) (size (- sx (* (+ dpml 0.1) 2)) sy sz )) output-efield-z))
)

(if no-metal? (save-flux "refl-flux" refl))

(display-fluxes trans  refl )
