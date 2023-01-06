d_dtau^2 t = 0
d_dtau^2 x = 0

t(tau) = a * tau + b
x(tau) = c * tau + d
y(tau) = e * tau + f

# irgendein affiner parameter

t^2 === x^2 + y^2 for all tau

a^2 tau^2 + 2 * a * b * tau + b^2

=

(c^2 + e^2) * tau^2 + 2 (cd + ef) tau + (d^2 + f^2)

-->
a^2 = c^2 + e^2
b^2 = d^2 + f^2

--> aus 6 unabh. Konstanten werden 4.

a*b = cd + ef

a^2 b^2 = (c^2 + e^2)(d^2 + f^2) = (cd + ef)^2 = (a*b)^2
--> Das ist eine Bestimmungsgleichung (sprich: eine Konstante weniger)

--> 3 unabhängige Konstanten (mindestens)


x(tau1) == xobs = c * tau1 + d
x(tau2) == xem  = c * tau2 + d

y analog
yobs = e * tau1 + f
yem  = e * tau2 + f

--> bestimmen mir klar e und f in abh. von xobs/yobs xem / yem

tau1 / tau2 = (xobs - d) / (xem - d) = (yobs - f) / (yem - f)

----> Eine Bestimmungsgleichung d(f) oder f(d)
---> eine Konstante weniger --> höchstens 2

xobs - d = c * tau1 ---- yem - f = e * tau2
--> c tau1 / e tau2 = (xobs - d) / (yem - f) --->

tau1 / tau2 = e / c * (xobs - d) / (yem - f) ==(analog)== c / e * (yobs - f) / (xem - d)

---> Eine Bestimmungsgl. ---> eine Konstante weniger

