@startuml

actor "uporabnik racunalnik" as  uporabnikr <<Zunanji>>
actor "uporabnik telefon" as  uporabnikt <<Zunanji>>
actor "administrator" as admin <<notranji>>
package WinccServer{
usecase prijavljanje

package UI{
usecase vpogled

}

package bazapodatkov{

}

}

vpogled ..> admin
uporabnikr --> prijavljanje
uporabnikt --> prijavljanje
prijavljanje -> vpogled
@enduml