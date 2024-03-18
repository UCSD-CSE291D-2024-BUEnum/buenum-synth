; Synthesize using only AND and NOT gates

(set-logic BV)

(define-fun customize ((a Bool) (b Bool)) Bool
  (or a b))

(synth-fun AIG ((a Bool) (b Bool) (c Bool)) Bool
           ((Start Bool ((and Start Start) (not Start) a b c))))

(declare-var a Bool)
(declare-var b Bool)
(declare-var c Bool)

(constraint
 (= (customize (customize a b) c) 
    (AIG a b c)
        ))


(check-synth)

; Solution:
;(define-fun AIG ((a Bool) (b Bool) (c Bool)) Bool
;(not (and (not a) (not b)))) 